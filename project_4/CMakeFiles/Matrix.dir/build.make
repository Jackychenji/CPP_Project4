# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jackydjl/project4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jackydjl/project4

# Include any dependencies generated for this target.
include CMakeFiles/Matrix.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Matrix.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Matrix.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Matrix.dir/flags.make

CMakeFiles/Matrix.dir/Matrix.c.o: CMakeFiles/Matrix.dir/flags.make
CMakeFiles/Matrix.dir/Matrix.c.o: Matrix.c
CMakeFiles/Matrix.dir/Matrix.c.o: CMakeFiles/Matrix.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jackydjl/project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/Matrix.dir/Matrix.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/Matrix.dir/Matrix.c.o -MF CMakeFiles/Matrix.dir/Matrix.c.o.d -o CMakeFiles/Matrix.dir/Matrix.c.o -c /home/jackydjl/project4/Matrix.c

CMakeFiles/Matrix.dir/Matrix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Matrix.dir/Matrix.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jackydjl/project4/Matrix.c > CMakeFiles/Matrix.dir/Matrix.c.i

CMakeFiles/Matrix.dir/Matrix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Matrix.dir/Matrix.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jackydjl/project4/Matrix.c -o CMakeFiles/Matrix.dir/Matrix.c.s

CMakeFiles/Matrix.dir/test.c.o: CMakeFiles/Matrix.dir/flags.make
CMakeFiles/Matrix.dir/test.c.o: test.c
CMakeFiles/Matrix.dir/test.c.o: CMakeFiles/Matrix.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jackydjl/project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/Matrix.dir/test.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/Matrix.dir/test.c.o -MF CMakeFiles/Matrix.dir/test.c.o.d -o CMakeFiles/Matrix.dir/test.c.o -c /home/jackydjl/project4/test.c

CMakeFiles/Matrix.dir/test.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Matrix.dir/test.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jackydjl/project4/test.c > CMakeFiles/Matrix.dir/test.c.i

CMakeFiles/Matrix.dir/test.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Matrix.dir/test.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jackydjl/project4/test.c -o CMakeFiles/Matrix.dir/test.c.s

# Object files for target Matrix
Matrix_OBJECTS = \
"CMakeFiles/Matrix.dir/Matrix.c.o" \
"CMakeFiles/Matrix.dir/test.c.o"

# External object files for target Matrix
Matrix_EXTERNAL_OBJECTS =

Matrix: CMakeFiles/Matrix.dir/Matrix.c.o
Matrix: CMakeFiles/Matrix.dir/test.c.o
Matrix: CMakeFiles/Matrix.dir/build.make
Matrix: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
Matrix: /usr/lib/x86_64-linux-gnu/libpthread.a
Matrix: CMakeFiles/Matrix.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jackydjl/project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable Matrix"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Matrix.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Matrix.dir/build: Matrix
.PHONY : CMakeFiles/Matrix.dir/build

CMakeFiles/Matrix.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Matrix.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Matrix.dir/clean

CMakeFiles/Matrix.dir/depend:
	cd /home/jackydjl/project4 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jackydjl/project4 /home/jackydjl/project4 /home/jackydjl/project4 /home/jackydjl/project4 /home/jackydjl/project4/CMakeFiles/Matrix.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Matrix.dir/depend
