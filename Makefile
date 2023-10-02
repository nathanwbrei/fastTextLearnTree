#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

CXX = c++
CXXFLAGS = -pthread -std=c++0x
OBJS = args.o dictionary.o matrix.o vector.o lomtree.o model.o utils.o fasttext.o
INCLUDES = -I. -I/opt/homebrew/include

opt: CXXFLAGS += -O3 -funroll-loops
opt: fasttext

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/args.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/dictionary.cc

matrix.o: src/matrix.cc src/matrix.h src/utils.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/matrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/vector.cc

lomtree.o: src/lomtree.cc src/lomtree.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/lomtree.cc

model.o: src/model.cc src/model.h src/args.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/model.cc

utils.o: src/utils.cc src/utils.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/utils.cc

fasttext.o: src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/fasttext.cc

fasttext: $(OBJS) src/fasttext.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) src/main.cc -o fasttext

clean:
	rm -rf *.o fasttext
