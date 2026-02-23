CXX = clang++
SRC_DIR = src
SRCS = $(shell find $(SRC_DIR) -name "*.cpp")

# Base object directories
BUILD_DIR = build
DEV_OBJDIR = $(BUILD_DIR)/dev
RELEASE_OBJDIR = $(BUILD_DIR)/release

CXXFLAGS_DEV = ...  -c
CXXFLAGS_RELEASE = ... -MMD -MP -c

SANITIZE = -fsanitize=address,undefined -fno-sanitize-recover=all

# Compiler flags
CXXFLAGS_DEV =	-I. \
				-fcolor-diagnostics \
				-fansi-escape-codes \
				-pedantic-errors \
                -Wall \
                -Weffc++ \
                -Wextra \
                -Wconversion \
                -Wsign-conversion \
                -Werror \
				-std=c++23 \
				-ggdb \
				-g \
				-O0 \
				$(SANITIZE) \
				-MMD -MP \
				-c

CXXFLAGS_RELEASE = 	-I. \
					-std=c++23 \
					-O3 \
					-DNDEBUG \
					-MMD -MP \
					-c

# Map sources to object files
DEV_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(DEV_OBJDIR)/%.o,$(SRCS))
RELEASE_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(RELEASE_OBJDIR)/%.o,$(SRCS))

# Remove main.o for test builds
DEV_OBJS_NO_MAIN = $(filter-out $(DEV_OBJDIR)/main.o,$(DEV_OBJS))
RELEASE_OBJS_NO_MAIN = $(filter-out $(RELEASE_OBJDIR)/main.o,$(RELEASE_OBJS))

# ========================
# Targets
# ========================

# Default dev build
dev: $(DEV_OBJS)
	clang++ $(DEV_OBJS) $(SANITIZE) -o main_dev.out

release: $(RELEASE_OBJS)
	clang++ $(RELEASE_OBJS) -o main_release.out

# Test build (debug)
test: $(DEV_OBJS_NO_MAIN) tests/test.cpp
	@mkdir -p $(dir $@)
	clang++ $(DEV_OBJS_NO_MAIN) tests/test.cpp -I. -std=c++23 -g -O0 $(SANITIZE) -o tests/test.out

# ========================
# Pattern rules for objects
# ========================

# Dev objects
$(DEV_OBJDIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_DEV) $< -o $@

# Release objects
$(RELEASE_OBJDIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_RELEASE) $< -o $@

# ========================
# Clean
# ========================
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) main_dev.out main_release.out tests/test.out

# Include auto-generated dependency files
-include $(DEV_OBJS:.o=.d)
-include $(RELEASE_OBJS:.o=.d)
