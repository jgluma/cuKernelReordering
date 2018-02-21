#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <thread>

using namespace std;

#define UNIFORM	1
#define NORMAL	2
#define EXPONENTIAL 3
#define POISSION 4

void escribirFichero(ofstream &fich, int nepoch, int distribution, float a, float b)
{
	switch(distribution)
	{
		case NORMAL:
		{
			
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    		const int nrolls=10000;
  			std::default_random_engine generator(seed);
  			std::normal_distribution<double> distribution(0.0,0.0);

  			

  			for (int i=0; i<nepoch; ++i) {
  				double number;
  				for (int i=0; i<nrolls; ++i)
  				{
  					number = distribution(generator);
    				if ((number>=0.0)&&(number<1.0))
    				{
    					number = number * 100;
    					break; //cout<< number << endl;//++p[int(number)];
    				}
    					
  				}


    			
    			fich << number << "\t";
  			}


			break;
		}

		case EXPONENTIAL:
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

			const int nrolls=10000;  // number of experiments
  			const int nstars=100;    // maximum number of stars to distribute
  			const int nintervals=10; // number of intervals

  			std::default_random_engine generator(seed);
  			std::exponential_distribution<double> distribution(3.5);

    		for(int e = 0; e < nepoch; e++)
    		{
    			double number;
    			for (int i=0; i<nrolls; ++i) {
    				number = distribution(generator);
    				if (number>=0.0 && number<1.0)
    				{
    					number = number *100;
    					break;
    				}
    			}

    			fich << number << "\t";
  			}



			break;
		}

		case POISSION:
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

  			const int nrolls=10000;  // number of experiments
  			std::default_random_engine generator(seed);
  			std::poisson_distribution<int> distribution(100);

  			int p[10]={};

  			for(int e = 0; e < nepoch; e++)
  			{
  				int number;

  				for (int i=0; i<nrolls; ++i) {
    				number = distribution(generator);
    				if (number>= 0 && number <1000) 
    				{
    					//cout << (float)number/1000 << endl;
    					break;
    					
    				}
  				}

  				
  				fich << (float)number/10 << "\t";
  			}

			break;
		}

		case UNIFORM:
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

  			const int nrolls=10000;  // number of experiments
  			std::default_random_engine generator(seed);
  			std::uniform_real_distribution<double> distribution(a,b);

  			
  			double number;
  			for(int e = 0; e < nepoch; e++)
  			{
  				 number = distribution(generator);

  				fich << number << "\t";
  			}

  				
  				
  			

			break;
		}
	}
}

int main(int argc, char *argv[])
{

	if(argc != 6)
	{
		cout << "Execute: <program> <nproducer> <nepoch> <distribution> <interval1> <interval2>" << endl;
	}
	int nproducer = atoi(argv[1]);
	int nepoch = atoi(argv[2]);
	int distribution = atoi(argv[3]);
	string str_interval1(argv[4]);
	string str_interval2(argv[5]);
	float a = atoi(argv[4])*1.0;
	float b = atoi(argv[5])*1.0;

	string name; 

	switch(distribution)
	{
		case NORMAL:
		{
			name = "times_normal_" + to_string(nproducer) + "p_" + to_string(nepoch) + "e.txt";
			break;
		}

		case EXPONENTIAL:
		{
			name = "times_exponential_" + to_string(nproducer) + "p_" + to_string(nepoch) + "e.txt";
			break;
		}

		case POISSION:
		{
			name = "times_poisson_" + to_string(nproducer) + "p_" + to_string(nepoch) + "e.txt";
			break;
		}

		case UNIFORM:
		{
			//name = "times_uniform_" + to_string(nproducer) + "p_" + to_string(nepoch) + "e.txt";
			name = "times_uniform_" + to_string(nproducer) + "p_" + to_string(nepoch) 
							+ "e_i" + str_interval1 + "-" + str_interval2 + ".txt";
			break;
		}
	}

	

	ofstream fich(name);

	for(int p = 0; p < nproducer; p++)
	{
		escribirFichero(fich, nepoch, distribution, a, b);
		fich << endl;
	}

	


	fich.close();

	


	
	return 1;
}
