#ifndef _BUFFER_H_
#define _BUFFER_H_

#include <iostream>
#include <vector>
#include <deque>
#include <atomic>

#define STOP	-1
#define EMPTY	-2
#define NOTASK	-3

using namespace std;

struct infoTask{
	int id_thread;
	int id_task;
};

class BufferTasks
{
private:
	/*int nproducer;
	deque<infoTask> *buffers;
	atomic<int> *produced_elements;

	deque<infoTask> pending_tasks;
	atomic<int> n_pending_tasks;*/

	deque<infoTask> buffer;
	atomic<int> produced_elements;

public:

	
	BufferTasks();
	void pushBack(infoTask task);
	void popFront(void);
	void getFront(infoTask &task);
	void getTask(infoTask &task, int idx);
	void printBuffer(void);
	void deleteSetTasks(vector<infoTask> &tasks_set);
	int getProducedElements(void);

};

#endif
