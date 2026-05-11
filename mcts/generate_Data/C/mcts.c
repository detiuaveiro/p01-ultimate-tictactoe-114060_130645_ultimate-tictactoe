#include "mcts.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Node* node_create(const UTTTGame* game, const Action* action) {
    Node* node = (Node*)malloc(sizeof(Node));
    if (!node) return NULL;
    node->game = uttt_clone(game);
    if (action) node->action = *action;
    else { node->action.x = 0; node->action.y = 0; node->action.symbol = 0; node->action.index = 0; }
    node->child_nodes = NULL;
    node->num_children = 0; node->num_visits = 0;
    node->num_X_wins = 0; node->num_O_wins = 0; node->num_draws = 0;
    return node;
}

// Create node taking ownership of the game (no extra clone).
Node* node_create_with_game(UTTTGame* game, const Action* action) {
    Node* node = (Node*)malloc(sizeof(Node));
    if (!node) { if (game) uttt_destroy(game); return NULL; }
    node->game = game;
    if (action) node->action = *action;
    else { node->action.x = 0; node->action.y = 0; node->action.symbol = 0; node->action.index = 0; }
    node->child_nodes = NULL;
    node->num_children = 0; node->num_visits = 0;
    node->num_X_wins = 0; node->num_O_wins = 0; node->num_draws = 0;
    return node;
}

void node_destroy(Node* node) {
    if (!node) return;
    if (node->child_nodes) {
        for (int i = 0; i < node->num_children; i++) node_destroy(node->child_nodes[i]);
        free(node->child_nodes);
    }
    if (node->game) uttt_destroy(node->game);
    free(node);
}

bool node_is_leaf(const Node* node) { return node && node->num_children == 0; }
bool node_is_terminal(const Node* node) { return node && uttt_check_game_over(node->game); }

void node_expand(Node* node) {
    if (!node || !node_is_leaf(node) || node_is_terminal(node)) return;
    int action_count;
    Action* actions = uttt_get_valid_actions(node->game, &action_count);
    if (action_count == 0) { free(actions); return; }

    node->child_nodes = (Node**)malloc(action_count * sizeof(Node*));
    for (int i = 0; i < action_count; i++) {
        UTTTGame* cloned = uttt_clone(node->game);
        uttt_process_move(cloned, actions[i].x, actions[i].y);
        node->child_nodes[i] = node_create_with_game(cloned, &actions[i]);
    }
    node->num_children = action_count;
    free(actions);
}

Tree* tree_create(const UTTTGame* game) {
    Tree* tree = (Tree*)malloc(sizeof(Tree));
    tree->root = node_create(game, NULL);
    return tree;
}

void tree_destroy(Tree* tree) {
    if (!tree) return;
    if (tree->root) node_destroy(tree->root);
    free(tree);
}

bool nodes_equal(const Node* a, const Node* b) { return uttt_is_equal_to(a->game, b->game); }

// Synchronize the tree with the current game state.
void tree_synchronize(Tree* tree, const UTTTGame* game) {
    if (!tree || !game) return;
    for (int i = 0; i < tree->root->num_children; i++) {
        if (uttt_is_equal_to(tree->root->child_nodes[i]->game, game)) {
            Node* old_root = tree->root;
            tree->root = tree->root->child_nodes[i];
            old_root->child_nodes[i] = NULL;  
            node_destroy(old_root);
            return;
        }
    }
    Node* old_root = tree->root;
    tree->root = node_create(game, NULL);
    node_destroy(old_root);
}

MonteCarloTreeSearch* mcts_create(const UTTTGame* game, int num_simulations, float exploration_strength) {
    MonteCarloTreeSearch* mcts = (MonteCarloTreeSearch*)malloc(sizeof(MonteCarloTreeSearch));
    mcts->tree = tree_create(game);
    mcts->num_simulations = num_simulations;
    mcts->exploration_strength = exploration_strength;
    return mcts;
}

void mcts_destroy(MonteCarloTreeSearch* mcts) {
    if (!mcts) return;
    if (mcts->tree) tree_destroy(mcts->tree);
    free(mcts);
}

void mcts_run(MonteCarloTreeSearch* mcts) {
    if (!mcts || !mcts->tree || !mcts->tree->root) return;
    int num_run_simulations = mcts->num_simulations - mcts->tree->root->num_visits;
    for (int i = 0; i < num_run_simulations; i++) {
        simulate(mcts->tree->root, mcts->exploration_strength);
    }
}

float value_function(Node* node) {
    if (!node || node->num_visits == 0) return 0.0f;
    int symbol = node->action.symbol;
    int num_wins = (symbol == 1) ? node->num_X_wins : node->num_O_wins;
    int num_losses = (symbol == 1) ? node->num_O_wins : node->num_X_wins;
    return (float)(num_wins - num_losses) / node->num_visits;
}

float uct_value(Node* node, int parent_visits, float exploration_strength) {
    if (!node) return -1e9f;
    if (node->num_visits == 0) return 1e9f;
    float exploitation = value_function(node);
    float exploration = exploration_strength * sqrtf(logf(parent_visits) / node->num_visits);
    return exploitation + exploration;
}

Node* select_leaf_node(Node* node, float exploration_strength, Node** path, int* path_len) {
    *path_len = 0;
    while (!node_is_leaf(node)) {
        path[*path_len] = node;
        (*path_len)++;
        float best_score = -1e9f;
        int best_child = -1;
        for (int i = 0; i < node->num_children; i++) {
            float score = uct_value(node->child_nodes[i], node->num_visits, exploration_strength);
            if (score > best_score) { best_score = score; best_child = i; }
        }
        if (best_child == -1) break;
        node = node->child_nodes[best_child];
    }
    path[*path_len] = node;
    (*path_len)++;
    return node;
}

void playout(Node* node, int* p1_wins, int* p2_wins, int* draws) {
    *p1_wins = 0; *p2_wins = 0; *draws = 0;
    if (!node) return;
    UTTTGame* game = uttt_clone(node->game);
    while (!uttt_check_game_over(game)) {
        int action_count;
        Action* actions = uttt_get_valid_actions(game, &action_count);
        if (action_count == 0) { free(actions); break; }
        int random_idx = rand() % action_count;
        uttt_process_move(game, actions[random_idx].x, actions[random_idx].y);
        free(actions);
    }
    uttt_who_is_winner(game);
    if (game->winner == 1) *p1_wins = 1;
    else if (game->winner == 2) *p2_wins = 1;
    else if (game->winner == 3) *draws = 1;
    uttt_destroy(game);
}

void backprop(Node** path, int path_len, int p1_wins, int p2_wins, int draws) {
    for (int i = 0; i < path_len; i++) {
        if (!path[i]) continue;
        path[i]->num_visits++;
        path[i]->num_X_wins += p1_wins;
        path[i]->num_O_wins += p2_wins;
        path[i]->num_draws += draws;
    }
}

void simulate(Node* node, float exploration_strength) {
    if (!node) return;
    Node* path[1000];
    int path_len;
    Node* leaf = select_leaf_node(node, exploration_strength, path, &path_len);
    if (!leaf) return;
    node_expand(leaf);
    int p1_wins, p2_wins, draws;
    playout(leaf, &p1_wins, &p2_wins, &draws);
    backprop(path, path_len, p1_wins, p2_wins, draws);
}

EvaluatedAction* node_get_evaluated_actions(const Node* node, int* count) {
    if (!node || !count || node_is_leaf(node) || node->num_visits == 0) {
        *count = 0; return NULL;
    }
    EvaluatedAction* actions = (EvaluatedAction*)malloc(node->num_children * sizeof(EvaluatedAction));
    for (int i = 0; i < node->num_children; i++) {
        Node* child = node->child_nodes[i];
        int symbol = child->action.symbol;
        actions[i].symbol = symbol;
        actions[i].index = child->action.index;
        actions[i].num_visits = child->num_visits;
        actions[i].num_wins = (symbol == 1) ? child->num_X_wins : child->num_O_wins;
        actions[i].draws = child->num_draws;
        actions[i].losses = (symbol == 1) ? child->num_O_wins : child->num_X_wins;
    }
    *count = node->num_children;
    return actions;
}

int** node_get_evaluated_state(const Node* node) {
    if (!node) return NULL;
    return uttt_get_state(node->game);
}