#include "batches.h"

using namespace std;


bool Batch::insertTask(int tid, int id_task, int batch_type)
{
	bool inserted = false;

	pair<map<int,int>::iterator,bool> ret;

	ret = batch.insert(pair<int,int>(tid, id_task));

	if(ret.second != false)
		inserted = true;
	
	return inserted;
}

void Batch::copyBatch(map<int,int> b)
{
	batch = b;
}

int Batch::getTamBatch(void)
{
	return batch.size();
}

void Batch::cleanBatch(void)
{
	
	batch.clear();
	
}



void Batch::printBatch()
{

	for (map<int,int>::iterator it=batch.begin(); it!=batch.end(); ++it)
	{
		cout << "\tProceso " << it->first << " - Tarea " << it->second << endl;
	}
}
