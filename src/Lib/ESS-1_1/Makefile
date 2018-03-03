CXXFLAGS=-O3

ess:    ess.cc quality_pyramid.o quality_box.o ess.o
	g++ $(CXXFLAGS) -D__MAIN__ -o ess ess.cc quality_pyramid.o quality_box.o 

ess.o:	
	g++ -c ess.cc
	
quality_pyramid.o:	
	g++ -c quality_pyramid.cc
	
quality_box.o:	
	g++ -c quality_box.cc

libs:	ess.cc quality_pyramid.o quality_box.o
	g++ $(CXXFLAGS) -fPIC -shared -Wl,-soname,libess.so -o libess.so ess.cc quality_pyramid.o quality_box.o -lc

test:   ess
	maxresults=4 ./ess 5 5 examples/test_corners.weight examples/test_corners.clst
	# Should look like this:
	# 1.39999997616 4 4 4 4 1.29999995232 4 0 4 1 1.20000004768 1 4 1 4 1.10000002384 1 1 1 1

examples: ess
	./ess 368 272 examples/cow.weight examples/cow.clst
	# Should look like this:
	# 10978.8779297 89 117 258 209
	
	./ess 151 101 examples/car-l1.weight examples/car.clst
        # 1.51183950901 39 56 118 77 
	
	numlevels=2 ./ess 151 101 examples/car-l2.weight examples/car.clst
	# 1.69141745567 33 55 115 77 

clean:
	rm -f ess *.o
