#include<iostream>
#include<ctime>
#include<vector>
#include<fstream>
#include<cmath>
#include<ctime>
using namespace std;

const double alpha = 0.9;
const double belta = 0.1;
const double r = 2;

const int num_ant = 10;
double p0 = -1;
const int iteration = 2000;

const int MAX = 1999999999;
const int N = 100;

double connection[N][N];
double points[N][2];
double pheromone[N][N] = { 0 };

struct ant
{
	vector<int> path;
	double cost = 0;
	bool visited[N] = { 0 };

	void visit(int i)
	{
		if (i < 0 || visited[i])
			return;
		if (path.size() > 0)
			cost += connection[path[path.size() - 1]][i];
		path.push_back(i);
		visited[i] = true;
	}

	void back()
	{
		cost += connection[path[path.size() - 1]][path[0]];
		path.push_back(path[0]);
	}

	void init()
	{
		path.clear();
		cost = 0;
		for (int i = 0; i < N; i++)
			visited[i] = false;
		visit(rand() % N);
	}

	bool operator<(const ant& a)const
	{
		return this->cost < a.cost;
	}
};

ant global_best;
ant ants[num_ant];

double dis(double x[2], double y[2])
{
	return pow(pow((x[0] - y[0]), 2) + pow((x[1] - y[1]), 2), 0.5);
}


void get_data()
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			connection[i][j] = -1;

	fstream data;
	data.open("kroA100.tsp", ios::in);
	int buffer;
	for (int i = 0; i < N; i++)
		data >> buffer >> points[i][0] >> points[i][1];

	for (int i = 0; i < N; i++)
		for (int j = i; j < N; j++)
			connection[i][j] = connection[j][i] = round(dis(points[i], points[j]));

	if (p0 == -1)
	{
		double len = 0;
		bool visit[N] = { 0 };
		int now = rand() % N;
		for (int i = 1; i < N; i++)
		{
			visit[now] = true;
			double min = MAX;
			int nex = -1;
			for (int j = 0; j < N; j++)
			{
				if (!visit[j] && connection[now][j] != -1)
				{
					if (connection[now][j] < min)
					{
						min = connection[now][j];
						nex = j;
					}
				}
			}
			if (nex != -1)
			{
				len += min;
				now = nex;
			}
			else
				break;
		}
		len += connection[now][0];
		p0 = 1 / (N * len);
	}
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			pheromone[i][j] = p0;
}

double priority(int i, int j)
{
	return pheromone[i][j] * pow((1 / connection[i][j]), r);
}

int find_next(ant an)
{
	bool explore = ((rand() % 10001) / 10000.0) > alpha;
	int aim = -2;
	int now = an.path.back();
	if (explore)
	{
		double to = 0;
		vector<double> probability;
		vector<int> city;
		for (int i = 0; i < N; i++)
		{
			if (connection[i][now] == -1 || an.visited[i])
				continue;
			double pri = priority(i, now);
			to += pri;
			probability.push_back(pri);
			city.push_back(i);
		}

		for (int i = 0; i < probability.size(); i++)
		{
			probability[i] /= to;
		}
		double p = (rand() % 10001) / 10000.0;
		for (int i = 0; i < city.size(); i++)
		{
			p -= probability[i];
			if (p <= 1e-14)
			{
				aim = city[i];
				break;
			}
		}
	}
	else {
		double max = -1;
		for (int i = 0; i < N; i++)
		{
			if (connection[i][now] == -1 || an.visited[i])
				continue;
			if (priority(i, now) > max)
			{
				max = priority(i, now);
				aim = i;
			}
		}
	}
	return aim;
}

void local_update(int i, int j)
{
	pheromone[i][j] = pheromone[j][i] = (1 - belta) * pheromone[i][j] + belta * p0;
}

void global_update(ant best)
{
	for (int i = 1; i < best.path.size(); i++)
	{
		pheromone[best.path[i]][best.path[i - 1]] = (1 - belta) * pheromone[best.path[i]][best.path[i - 1]] + belta * (1 / best.cost);
		pheromone[best.path[i - 1]][best.path[i]] = pheromone[best.path[i]][best.path[i - 1]];
	}
}

void print(int it, ant re)
{
	cout << "iteration: " << it << "\ntrack: " << re.path[0];
	for (int i = 1; i < re.path.size(); i++)
		cout << "->" << re.path[i];
	cout << "\ntotal_leangth: " << re.cost << "\n" << endl;
}

int main()
{
	time_t begin, end;
	begin = clock();
	srand(time(NULL));
	get_data();
	global_best.cost = MAX;
	int best_it;

	for (int it = 0; it < iteration; it++)
	{
		for (int i = 0; i < num_ant; i++)
			ants[i].init();
		for (int i = 1; i < N; i++)
		{
			for (int j = 0; j < num_ant; j++)
			{
				int nex = find_next(ants[j]);
				if (nex < 0)
					cout << "mistake!" << endl;
				local_update(ants[j].path[ants[j].path.size() - 1], nex);
				ants[j].visit(nex);
			}
		}
		for (int i = 0; i < num_ant; i++)
		{
			local_update(ants[i].path[0], ants[i].path[N - 1]);
			ants[i].back();
		}

		ant local_best = ants[0];
		for (int i = 1; i < num_ant; i++)
			if (ants[i] < local_best)
				local_best = ants[i];
		if (local_best < global_best)
		{
			global_best = local_best;
			best_it = it;
		}

		// print(it, local_best);
		global_update(local_best);
		global_update(global_best);
	}
	end = clock();
	cout << "*****************************\n"
		<< "The best is:";
	print(best_it, global_best);
	cout<<"run time: "<<double(end - begin) / 1000.0<<"s"<<endl;

	return 0;
}