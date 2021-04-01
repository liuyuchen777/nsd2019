CXX = g++

question_one: question_one.o
	$(CXX) $^ -o $@
question_one.o: question_one.cpp 
	$(CXX) -c $< -o $@
run:
	./question_one
clean:
	rm question_one.o question_one