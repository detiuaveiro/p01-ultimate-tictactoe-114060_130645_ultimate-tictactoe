#include "game.h"
#include <stdlib.h>
#include <string.h>

UTTTGame* uttt_create(void) {
    UTTTGame* game = (UTTTGame*)malloc(sizeof(UTTTGame));
    if (game) uttt_reset(game);
    return game;
}

void uttt_destroy(UTTTGame* game) {
    if (game) free(game);
}

void uttt_reset(UTTTGame* game) {
    if (!game) return;
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            game->board[i][j] = 0;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            game->macro_board[i][j] = 0;

    game->active_macro[0] = -1;
    game->active_macro[1] = -1;
    game->current_turn = 1;
    game->winner = 0;
}

UTTTGame* uttt_clone(const UTTTGame* game) {
    if (!game) return NULL;
    UTTTGame* cloned = (UTTTGame*)malloc(sizeof(UTTTGame));
    if (!cloned) return NULL;
    memcpy(cloned, game, sizeof(UTTTGame));
    return cloned;
}

int uttt_check_3x3_win(const int board[9][9], int start_x, int start_y) {
    for (int i = 0; i < 3; i++) {
        if (board[start_y + i][start_x] != 0 &&
            board[start_y + i][start_x] == board[start_y + i][start_x + 1] &&
            board[start_y + i][start_x] == board[start_y + i][start_x + 2])
            return board[start_y + i][start_x];
    }
    for (int i = 0; i < 3; i++) {
        if (board[start_y][start_x + i] != 0 &&
            board[start_y][start_x + i] == board[start_y + 1][start_x + i] &&
            board[start_y][start_x + i] == board[start_y + 2][start_x + i])
            return board[start_y][start_x + i];
    }
    if (board[start_y][start_x] != 0 &&
        board[start_y][start_x] == board[start_y + 1][start_x + 1] &&
        board[start_y][start_x] == board[start_y + 2][start_x + 2])
        return board[start_y][start_x];

    if (board[start_y + 2][start_x] != 0 &&
        board[start_y + 2][start_x] == board[start_y + 1][start_x + 1] &&
        board[start_y + 2][start_x] == board[start_y][start_x + 2])
        return board[start_y + 2][start_x];

    return 0;
}

bool uttt_is_3x3_full(const int board[9][9], int start_x, int start_y) {
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (board[start_y + y][start_x + x] == 0) return false;
        }
    }
    return true;
}

Action* uttt_get_valid_actions(const UTTTGame* game, int* count) {
    if (!game || !count) return NULL;
    *count = 0;
    Action* actions = (Action*)malloc(81 * sizeof(Action));
    if (!actions) return NULL;
    if (game->winner != 0) return actions;

    for (int y = 0; y < 9; y++) {
        for (int x = 0; x < 9; x++) {
            int my = y / 3;
            int mx = x / 3;
            if (game->macro_board[my][mx] != 0) continue;
            if (game->board[y][x] != 0) continue;
            if (game->active_macro[0] != -1) {
                if (game->active_macro[0] != my || game->active_macro[1] != mx) continue;
            }
            actions[*count].x = x;
            actions[*count].y = y;
            actions[*count].symbol = game->current_turn;
            actions[*count].index = y * 9 + x;
            (*count)++;
        }
    }
    return actions;
}

// Helper to detect a winner in the macro board.
int check_macro_winner(const int macro[3][3]) {
    for (int i = 0; i < 3; i++) {
        if (macro[i][0] != 0 && macro[i][0] == macro[i][1] && macro[i][0] == macro[i][2]) return macro[i][0];
        if (macro[0][i] != 0 && macro[0][i] == macro[1][i] && macro[0][i] == macro[2][i]) return macro[0][i];
    }
    if (macro[0][0] != 0 && macro[0][0] == macro[1][1] && macro[0][0] == macro[2][2]) return macro[0][0];
    if (macro[2][0] != 0 && macro[2][0] == macro[1][1] && macro[2][0] == macro[0][2]) return macro[2][0];
    return 0;
}

// Update winner state based on the macro board.
void uttt_who_is_winner(UTTTGame* game) {
    if (!game) return;
    int winner = check_macro_winner(game->macro_board);
    bool is_draw = true;
    for(int i=0; i<3; i++) 
        for(int j=0; j<3; j++) 
            if(game->macro_board[i][j] == 0) is_draw = false;

    if (winner == 1 || winner == 2) game->winner = winner;
    else if (is_draw) game->winner = 3;
}

// Check if the game is over based on winner or full macro board.
bool uttt_check_game_over(const UTTTGame* game) {
    if (!game) return false;
    if (game->winner != 0) return true;
    int winner = check_macro_winner(game->macro_board);
    if (winner != 0) return true;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            if(game->macro_board[i][j] == 0) return false;
    return true;
}
bool uttt_process_move(UTTTGame* game, int x, int y) {
    if (!game || game->winner != 0) return false;
    int action_count;
    Action* valid = uttt_get_valid_actions(game, &action_count);
    
    bool is_valid = false;
    for (int i = 0; i < action_count; i++) {
        if (valid[i].x == x && valid[i].y == y) {
            is_valid = true;
            break;
        }
    }
    free(valid);
    if (!is_valid) return false;

    game->board[y][x] = game->current_turn;
    int my = y / 3, mx = x / 3;
    int micro_y = y % 3, micro_x = x % 3;

    int local_winner = uttt_check_3x3_win((const int(*)[9])game->board, mx * 3, my * 3);
    if (local_winner) {
        game->macro_board[my][mx] = local_winner;
    } else if (uttt_is_3x3_full((const int(*)[9])game->board, mx * 3, my * 3)) {
        game->macro_board[my][mx] = 3;
    }

    int next_my = micro_y, next_mx = micro_x;
    if (game->macro_board[next_my][next_mx] != 0) {
        game->active_macro[0] = -1;
        game->active_macro[1] = -1;
    } else {
        game->active_macro[0] = next_my;
        game->active_macro[1] = next_mx;
    }

    uttt_who_is_winner(game);
    if (game->winner == 0) game->current_turn = 3 - game->current_turn;
    return true;
}

bool uttt_is_equal_to(const UTTTGame* a, const UTTTGame* b) {
    if (!a || !b) return false;
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            if (a->board[i][j] != b->board[i][j]) return false;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (a->macro_board[i][j] != b->macro_board[i][j]) return false;

    if (a->active_macro[0] != b->active_macro[0]) return false;
    if (a->active_macro[1] != b->active_macro[1]) return false;
    if (a->current_turn != b->current_turn) return false;
    if (a->winner != b->winner) return false;
    return true;
}

int** uttt_get_state(const UTTTGame* game) {
    if (!game) return NULL;
    int** state = (int**)malloc(9 * sizeof(int*));
    if (!state) return NULL;
    for (int i = 0; i < 9; i++) {
        state[i] = (int*)malloc(9 * sizeof(int));
        memcpy(state[i], game->board[i], 9 * sizeof(int));
    }
    return state;
}