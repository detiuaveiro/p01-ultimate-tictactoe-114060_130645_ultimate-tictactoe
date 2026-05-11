CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -std=c99
LDFLAGS = -lm
OMPFLAGS = -fopenmp

TARGET = orchestrator
TEST_TARGET = test_runner
SOURCES = orchestrator.c
ORCH_SOURCES = orchestrator.c game.c mcts.c
OBJECTS = $(SOURCES:.c=.o)
ORCH_OBJECTS = $(ORCH_SOURCES:.c=.o)
HEADERS = game.h mcts.h

all: $(TARGET)

$(TARGET): $(ORCH_OBJECTS)
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)

orchestrator.o: CFLAGS += $(OMPFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

run-orchestrator: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(OBJECTS) $(ORCH_OBJECTS) $(TARGET)

.PHONY: all clean test run-orchestrator
