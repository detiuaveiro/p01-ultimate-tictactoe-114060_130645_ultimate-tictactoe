#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <limits.h>
#include <getopt.h>
#include <omp.h> 

#include "game.h"
#include "mcts.h"

// Per-game history (UTTT has at most 81 moves).
typedef struct {
    int step;
    int player_turn;
    int state[9][9];       // 9x9 board
    int num_visits[81];    // MCTS visits per cell
} StepData;

static void print_usage(const char* prog_name) {
    printf("Usage:\n");
    printf("  %s [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  -g, --total-games N       Total games (default: 10000)\n");
    printf("  -s, --num-simulations N   MCTS simulations per move (default: 50000)\n");
    printf("  -w, --num-workers N       OpenMP workers (default: 8)\n");
    printf("  -b, --seed N              Base seed (default: 42)\n");
    printf("  -o, --output-dir PATH     Output folder (default: raw_data)\n");
    printf("  -h, --help                Show this help\n\n");
    printf("Example:\n");
    printf("  %s --total-games 1000 --num-simulations 10000 --num-workers 4 --seed 42 --output-dir raw_data\n", prog_name);
}

static int parse_positive_int(const char* value, const char* arg_name, int* out) {
    char* end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        fprintf(stderr, "Error: invalid value for %s: '%s'\n", arg_name, value);
        return 0;
    }
    if (parsed <= 0 || parsed > INT_MAX) {
        fprintf(stderr, "Error: %s must be between 1 and %d.\n", arg_name, INT_MAX);
        return 0;
    }

    *out = (int)parsed;
    return 1;
}

static int parse_int_arg(const char* value, const char* arg_name, int* out) {
    char* end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        fprintf(stderr, "Error: invalid value for %s: '%s'\n", arg_name, value);
        return 0;
    }
    if (parsed < INT_MIN || parsed > INT_MAX) {
        fprintf(stderr, "Error: %s is out of int range.\n", arg_name);
        return 0;
    }

    *out = (int)parsed;
    return 1;
}

// Write one step to JSONL.
void write_step_to_jsonl(FILE *file, StepData *data, int final_result) {
    fprintf(file, "{\"step\": %d, \"player_turn\": %d, \"state\": {\"state\": [", data->step, data->player_turn);
    
    // Write 9x9 matrix
    for (int y = 0; y < 9; y++) {
        fprintf(file, "[");
        for (int x = 0; x < 9; x++) {
            fprintf(file, "%d%s", data->state[y][x], (x == 8) ? "" : ", ");
        }
        fprintf(file, "]%s", (y == 8) ? "" : ", ");
    }
    fprintf(file, "]}, \"actions\": [");

    // Write actions (only those with visits to save space)
    int first_action = 1;
    for (int i = 0; i < 81; i++) {
        if (data->num_visits[i] > 0) {
            if (!first_action) fprintf(file, ", ");
            fprintf(file, "{\"index\": %d, \"num_visits\": %d}", i, data->num_visits[i]);
            first_action = 0;
        }
    }

    fprintf(file, "], \"final_game_result\": %d}\n", final_result);
}


// Worker function (executed by each CPU core).
void worker_task(int worker_id, int games_to_play, int base_seed, int num_simulations, const char* output_dir) {
    char filename[256];
    sprintf(filename, "%s/dataset_w%d.jsonl", output_dir, worker_id);
    
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("[Worker %d] Failed to create file.\n", worker_id);
        return;
    }

    printf("[Worker %d] Started. Writing to '%s'\n", worker_id, filename);

    StepData history[81]; 

    for (int i = 0; i < games_to_play; i++) {
        int current_seed = base_seed + (worker_id * 100000) + i;
        srand(current_seed);

        // Initialize the game
        UTTTGame* g = uttt_create();
        int step = 0;

        // Game loop
        while (!uttt_check_game_over(g)) {
            
            // Run MCTS
            MonteCarloTreeSearch* mcts = mcts_create(g, num_simulations, 1.414f);
            mcts_run(mcts);

            // Save step data
            history[step].step = step + 1;
            history[step].player_turn = g->current_turn;
            
            // Copy board state
            for (int y = 0; y < 9; y++) {
                for (int x = 0; x < 9; x++) {
                    history[step].state[y][x] = g->board[y][x];
                }
            }

            // Reset visit counts
            for(int k = 0; k < 81; k++) {
                history[step].num_visits[k] = 0;
            }

            // Read evaluated actions from the root
            int num_actions = 0;
            EvaluatedAction* actions = node_get_evaluated_actions(mcts->tree->root, &num_actions);
            
            int best_index = -1;
            int max_visits = -1;

            if (actions != NULL) {
                for (int k = 0; k < num_actions; k++) {
                    history[step].num_visits[actions[k].index] = actions[k].num_visits;
                    
                    // Track the most visited move
                    if (actions[k].num_visits > max_visits) {
                        max_visits = actions[k].num_visits;
                        best_index = actions[k].index;
                    }
                }
                free(actions);
            }

            // Apply the chosen move
            if (best_index != -1) {
                int move_x = best_index % 9;
                int move_y = best_index / 9;
                uttt_process_move(g, move_x, move_y);
            } else {
                break;
            }
            
            // Clean up MCTS
            mcts_destroy(mcts); 
            
            step++;
        }

        // Game over: write JSONL
        int final_winner = g->winner;
        for (int s = 0; s < step; s++) {
            write_step_to_jsonl(file, &history[s], final_winner);
        }
        fflush(file); 

        uttt_destroy(g);

        printf("[Worker %d] Game %d/%d saved. (Winner: P%d, Moves: %d)\n", 
               worker_id, i + 1, games_to_play, final_winner, step);
    }

    fclose(file);
    printf("[Worker %d] Finished.\n", worker_id);
}


// Main orchestrator.
int main(int argc, char* argv[]) {
    int total_games = 10000;
    int num_simulations = 50000;
    int num_workers = 8;
    int base_seed = 42;
    const char* output_dir = "raw_data";

    static struct option long_options[] = {
        {"total-games", required_argument, 0, 'g'},
        {"num-simulations", required_argument, 0, 's'},
        {"num-workers", required_argument, 0, 'w'},
        {"seed", required_argument, 0, 'b'},
        {"output-dir", required_argument, 0, 'o'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "g:s:w:b:o:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'g':
                if (!parse_positive_int(optarg, "--total-games", &total_games)) return 1;
                break;
            case 's':
                if (!parse_positive_int(optarg, "--num-simulations", &num_simulations)) return 1;
                break;
            case 'w':
                if (!parse_positive_int(optarg, "--num-workers", &num_workers)) return 1;
                break;
            case 'b':
                if (!parse_int_arg(optarg, "--seed", &base_seed)) return 1;
                break;
            case 'o':
                output_dir = optarg;
                if (output_dir[0] == '\0') {
                    fprintf(stderr, "Error: --output-dir cannot be empty.\n");
                    return 1;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (optind < argc) {
        fprintf(stderr, "Error: unexpected argument(s):");
        for (int i = optind; i < argc; i++) {
            fprintf(stderr, " %s", argv[i]);
        }
        fprintf(stderr, "\n");
        print_usage(argv[0]);
        return 1;
    }

    // Create output directory
    #if defined(_WIN32)
        mkdir(output_dir);
    #else
        mkdir(output_dir, 0777); 
    #endif

    printf("=== STARTING PARALLEL GENERATION (C/OpenMP) ===\n");
    printf("Target: %d games split across %d workers.\n\n", total_games, num_workers);
    printf("Config: simulations=%d, seed=%d, output='%s'\n\n", num_simulations, base_seed, output_dir);

    int base_games = total_games / num_workers;
    int remainder = total_games % num_workers;

    // OpenMP splits the loop across CPU cores.
    #pragma omp parallel for num_threads(num_workers)
    for (int w = 0; w < num_workers; w++) {
        int games_for_this_worker = base_games + ((w == num_workers - 1) ? remainder : 0);
        
        worker_task(w, games_for_this_worker, base_seed, num_simulations, output_dir);
    }

    printf("\n=== DONE ===\n");
    printf("All files saved in '%s'.\n", output_dir);

    return 0;
}