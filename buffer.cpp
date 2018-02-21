#include "buffer.h"

BufferTasks::BufferTasks()
{
	
	produced_elements.store(0);
}

void BufferTasks::pushBack(infoTask task)
{
	buffer.push_back(task);
	//Increment the produced elements in the buffer
	produced_elements.fetch_add(1);
}


void BufferTasks::popFront(void)
{
	if(produced_elements.load() != 0)
	{
		buffer.pop_front();
		produced_elements.fetch_add(-1);
	}

	
}

void BufferTasks::getFront(infoTask &task)
{
	if(produced_elements.load() != 0)
	{
		task.id_thread = buffer.at(0).id_thread;
		task.id_task = buffer.at(0).id_task;
	}
}


void BufferTasks::printBuffer()
{
	for(int t = 0; t < buffer.size(); t++)
		cout << "\t\tTid: " << buffer.at(t).id_thread << " - Task: " << buffer.at(t).id_task << endl;

}

int BufferTasks::getProducedElements(void)
{
	return produced_elements.load();
}

void BufferTasks::deleteSetTasks(vector<infoTask> &tasks_set)
{

	for(int z = 0; z < tasks_set.size(); z++)
	{
		for(int t = 0; t < getProducedElements(); t++)
		{
			if(tasks_set[z].id_thread == buffer.at(t).id_thread
				&& tasks_set[z].id_task == buffer.at(t).id_task)
			{
				buffer.erase(buffer.begin() + t);
				produced_elements.fetch_add(-1);
				tasks_set[z].id_thread = -1;
				tasks_set[z].id_task = -1;
			}
		}
	}

}



void BufferTasks::getTask(infoTask &task, int idx)
{
	task.id_thread = buffer.at(idx).id_thread;
	task.id_task = buffer.at(idx).id_task;
}


