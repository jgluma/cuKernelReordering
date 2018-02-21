#include "batch.h"

using namespace std;

Batch::Batch(int tam)
{
	max_tam_batch = tam;
}

bool Batch::insertTask(int tid, int id_task)
{
	bool inserted = false;

	pair<map<int,int>::iterator,bool> ret;

	if(getTamBatch() < max_tam_batch)
	{
		ret = batch.insert(pair<int,int>(tid, id_task));

		if(ret.second != false)
			inserted = true;
	}
	
	return inserted;
}

void Batch::copyBatch(Batch b)
{
	cleanBatch();

	for(int t = 0; t < b.getTamBatch(); t++)
	{
		int tid;
		int id_task;

		b.getTaskBatch(tid, id_task, t);

		insertTask(tid, id_task);
	}

	
}

int Batch::getTamBatch(void)
{
	return batch.size();
}

void Batch::cleanBatch(void)
{
	
	batch.clear();
	
}

int Batch::getProcessTaskBatch(int tid)
{
	map<int,int>::iterator it = batch.find(tid);

	return it->second;
}
void Batch::getTaskBatch(int &tid, int &id_task, int idx)
{
	map<int,int>::iterator it = batch.begin();

	int i = 0;
	while( i != idx)
	{
		it++;
		i++;
	}

	tid = it->first;
	id_task = it->second;
}

void Batch::printBatch()
{
	cout << "********** BATCH **********" << endl;
	for (map<int,int>::iterator it=batch.begin(); it!=batch.end(); ++it)
	{
		cout << "\tProceso " << it->first << " - Tarea " << it->second << endl;
	}
	cout << "********************" << endl;
}
