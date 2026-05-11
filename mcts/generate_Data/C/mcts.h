#ifndef MCTS_H
#define MCTS_H

#include "game.h"
#include <stdint.h>

typedef struct Node {
    UTTTGame* game;
    Action action;
    struct Node** child_nodes;
    int num_children;
    int num_visits;
    int num_X_wins;
    int num_O_wins;
    int num_draws;
} Node;

typedef struct { Node* root; } Tree;

typedef struct {
    Tree* tree;
    int num_simulations;
    float exploration_strength;
} MonteCarloTreeSearch;

Node* node_create(const UTTTGame* game, const Action* action);
Node* node_create_with_game(UTTTGame* game, const Action* action);
void node_destroy(Node* node);
bool node_is_leaf(const Node* node);
bool node_is_terminal(const Node* node);
void node_expand(Node* node);

Tree* tree_create(const UTTTGame* game);
void tree_destroy(Tree* tree);
void tree_synchronize(Tree* tree, const UTTTGame* game);

MonteCarloTreeSearch* mcts_create(const UTTTGame* game, int num_simulations, float exploration_strength);
void mcts_destroy(MonteCarloTreeSearch* mcts);
void mcts_run(MonteCarloTreeSearch* mcts);

void simulate(Node* node, float exploration_strength);
void playout(Node* node, int* p1_wins, int* p2_wins, int* draws);
void backprop(Node** path, int path_len, int p1_wins, int p2_wins, int draws);
Node* select_leaf_node(Node* node, float exploration_strength, Node** path, int* path_len);
float uct_value(Node* node, int parent_visits, float exploration_strength);
float value_function(Node* node);

typedef struct {
    int symbol;
    int index;
    int num_visits;
    int num_wins;
    int draws;
    int losses;
} EvaluatedAction;

EvaluatedAction* node_get_evaluated_actions(const Node* node, int* count);
int** node_get_evaluated_state(const Node* node);

#endif // MCTS_H