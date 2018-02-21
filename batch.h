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
  		int max_tam_batch;

public:
	Batch(int tam);
	bool insertTask(int tid, int id_task);
	int getTamBatch(void);
	void getTaskBatch(int &tid, int &id_task, int idx);
	int getProcessTaskBatch(int tid);
	void copyBatch(Batch b);
	void cleanBatch(void);
	void printBatch(void);
};

#endif
