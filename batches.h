#ifndef _BATCHES_H_
#define _BATCHES_H_
#include <iostream>
#include <map>
#include <iterator>
#include <atomic>



using namespace std;

class Batch
{
private:
		//Containers for the batch
  		map<int, int> batch;

public:
	bool insertTask(int tid, int id_task, int batch_type);
	int getTamBatch(void);
	void copyBatch(map<int,int> b);
	void cleanBatch(void);
	void printBatch(void);
};

#endif
