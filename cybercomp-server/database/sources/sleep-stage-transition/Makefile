all: mindcurrent

mindcurrent: main.cpp CellSyn.cpp CellSyn.h currents.cpp currents.h io.cpp io.h network.h network.cpp
	g++ -g -O3 -lm -Wall -fopenmp  main.cpp CellSyn.cpp io.cpp currents.cpp network.cpp -o $(current_stim)
	 #icpc -O3 -Wall -fopenmp main.cpp CellSyn.cpp io.cpp currents.cpp network.cpp -o $(current_stim)

#sanity check whether current version of code changes output (compared to previously stored test/.files)
check: mindcurrent
	./$(current_stim) $(current_params) test $(connection_info)
	cd test; for f in *; do diff -u $$f .$$f; done

check-prepare: mindcurrent
	./$(current_stim) $(current_params) test $(connection_info)
	cd test; for f in *; do cp -f  $$f .$$f; done

run: mindcurrent
	./$(current_stim) $(current_params) out $(connection_info)

doxy:
	doxygen ./docs/Doxyfile

clean: 
	-rm ./$(current_stim) 

network:
	g++ -O2 generate_network.cpp -o generate_network
	./generate_network $(network_config) $(mri_network) $(3D_subnet) $(3D_distance)> $(connection_info)



