#ifndef GAME_H
#define GAME_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int x;
    int y;
    int symbol;
    int index;
} Action;

typedef struct {
    int board[9][9];           // 0=Empty, 1=P1, 2=P2
    int macro_board[3][3];     // 0=Ongoing, 1=P1 Win, 2=P2 Win, 3=Draw
    int active_macro[2];       // [my, mx] or [-1, -1] for None
    int current_turn;          // 1 or 2
    int winner;                // 0=Ongoing, 1=P1 Win, 2=P2 Win, 3=Draw
} UTTTGame;

UTTTGame* uttt_create(void);
void uttt_destroy(UTTTGame* game);
void uttt_reset(UTTTGame* game);
UTTTGame* uttt_clone(const UTTTGame* game);

Action* uttt_get_valid_actions(const UTTTGame* game, int* count);
bool uttt_process_move(UTTTGame* game, int x, int y);
void uttt_who_is_winner(UTTTGame* game);
bool uttt_check_game_over(const UTTTGame* game);

int uttt_check_3x3_win(const int board[9][9], int start_x, int start_y);
bool uttt_is_3x3_full(const int board[9][9], int start_x, int start_y);
bool uttt_is_next_symbol_X(const UTTTGame* game);
bool uttt_is_equal_to(const UTTTGame* a, const UTTTGame* b);
int** uttt_get_state(const UTTTGame* game);

#endif // GAME_H