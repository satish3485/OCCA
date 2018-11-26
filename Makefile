# The MIT License (MIT)
#
# Copyright (c) 2014-2018 Satish Kumar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../../../scripts/Makefile
else
  include ${OCCA_DIR}/scripts/Makefile
endif

# #---[ COMPILATION ]-------------------------------
headers = $(wildcard $(incPath)/*.hpp) $(wildcard $(incPath)/*.tpp)
sources = $(wildcard $(srcPath)/*.cpp)

objects = $(subst $(srcPath)/,$(objPath)/,$(sources:.cpp=.o))

flags += -I$(PROJ_DIR)


${PROJ_DIR}/dot_producer: $(objects) $(headers) ${PROJ_DIR}/dot_producer.cpp ${PROJ_DIR}/cpu.hpp ${PROJ_DIR}/dense_matrix.hpp ${PROJ_DIR}/multi_grid.hpp ${PROJ_DIR}/sparse_matrix.hpp
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/dot_producer $(flags) $(objects) ${PROJ_DIR}/cpu.cpp ${PROJ_DIR}/dense_matrix.cpp ${PROJ_DIR}/multi_grid.cpp ${PROJ_DIR}/sparse_matrix.cpp ${PROJ_DIR}/dot_producer.cpp -L${OCCA_DIR}/lib $(paths) $(linkerFlags)

# ${PROJ_DIR}/dot_producer: $(objects) $(headers) ${PROJ_DIR}/dot_producer.cpp
# 	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/dot_producer $(flags) $(objects) ${PROJ_DIR}/dot_producer.cpp -L${OCCA_DIR}/lib $(paths) $(linkerFlags)

# $(objPath)/%.o:$(srcPath)/%.cpp $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.tpp)))
# 	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(objPath)/*;
	rm -f $(PROJ_DIR)/dot_producer


