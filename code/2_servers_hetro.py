# imports
import simpy
import numpy as np
import sys
import pandas as pd
import os
import pickle as pkl
# from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
# from scipy.special import factorial
import time
from tqdm import tqdm
from scipy.special import factorial
from scipy.linalg import expm, sinm, cosm

is_print = False


def ser_moment_n(s, A, mom):
    '''
    ser_moment_n
    :param s:
    :param A:
    :param mom:
    :return:
    '''
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) * factorial(mom) * np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False


def compute_first_n_moments(s, A, n=3):
    '''
    compute_first_n_moments
    :param s:
    :param A:
    :param n:
    :return:
    '''
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list


## Defining a new class: of customers
class Customer:
    def __init__(self, p_id, arrival_time, type_cust):
        self.id = p_id
        self.arrival_time = arrival_time
        self.type = type_cust


is_print = False


class N_Queue_single_station:

    def __init__(self,  sim_time, num_stations, services, arrivals_norm, num_servers):

        self.env = simpy.Environment()  # initializing simpy enviroment
        # Defining a resource with capacity 1
        self.end_time = sim_time  # The time simulation terminate
        self.id_current = 1  # keeping track of cusotmers id
        # an event can one of three: (1. arrival, 2. entering service 3. service completion)
        self.num_stations = num_stations

        self.server_occupy = np.array([False, False])
        self.servers = []
        self.df_waiting_times = pd.DataFrame([])  # is a dataframe the holds all information waiting time
        self.num_cust_durations = []
        self.df_waiting_times = []
        self.server = []
        self.last_event_time = []  # the time of the last event -
        self.num_cust_sys = []  # keeping track of number of customers in the system
        self.df_events = []  # is a dataframe the holds all information of the queue dynamic:
        self.last_depart = []
        self.inter_departures = {}
        # self.sojourn_times = []
        self.busy_times = [0,0]

        self.services = services
        self.arrivals = arrivals_norm

        for station in range(num_stations):
            self.servers.append(simpy.Resource(self.env, capacity=num_servers))
            self.num_cust_durations.append(
                np.zeros(500))  ## the time duration of each each state (state= number of cusotmers in the system)
            self.df_waiting_times.append(pd.DataFrame([]))  # is a dataframe the holds all information waiting time
            self.num_cust_sys.append(0)
            self.last_event_time.append(0)
            self.df_events.append(pd.DataFrame([]))
            self.last_depart.append(0)
            self.inter_departures[station] = []

    def run(self):

        station = 0
        self.env.process(self.customer_arrivals(station))  ## Initializing a process
        self.env.run(until=self.end_time)  ## Running the simulaiton until self.end_time

    def update_new_row(self, customer, event, station):

        new_row = {'Event': event, 'Time': self.env.now, 'Customer': customer.id,
                   'Queue lenght': len(self.servers[station].queue), 'System lenght': self.num_cust_sys[station],
                   'station': station}

        self.df_events[station] = pd.concat([self.df_events[station], pd.DataFrame([new_row])], ignore_index=True)

    #########################################################
    ################# Service block #########################
    #########################################################

    def service(self, customer, station):

        tot_time = self.env.now - self.last_event_time[station]
        self.num_cust_durations[station][self.num_cust_sys[station]] += tot_time
        self.num_cust_sys[station] += 1
        self.last_event_time[station] = self.env.now

        with self.servers[station].request() as req:


            yield req

            if self.server_occupy.sum() == 0:
                server_type = np.random.randint(2)


            elif self.server_occupy.sum() == 1:
                server_type = self.server_occupy.argmin()

            else:
                print('Stop')
            self.server_occupy[server_type] = True

            start_busy_time = self.env.now
            ind_ser = np.random.randint(self.services[server_type].shape[0])
            yield self.env.timeout(self.services[server_type][ind_ser])

            self.server_occupy[server_type] = False
            self.busy_times[server_type] += self.env.now - start_busy_time



            tot_time = self.env.now - self.last_event_time[station]  # keeping track of the last event
            self.num_cust_durations[station][
                self.num_cust_sys[station]] += tot_time  # Since the number of customers in the system changes
            # we compute how much time the system had this number of customers

            self.num_cust_sys[station] -= 1  # updating number of cusotmers in the system
            self.last_event_time[station] = self.env.now

            sojourn_time = self.env.now.item() -  customer.arrival_time.item()
            # self.sojourn_times.append(sojourn_time)

    #########################################################
    ################# Arrival block #########################
    #########################################################

    def customer_arrivals(self, station):

        while True:

            # ind_ser = np.random.randint(self.arrivals.shape[0])
            self.id_current += 1
            yield self.env.timeout(self.arrivals[self.id_current%self.arrivals.shape[0]])

            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id, arrival_time, 1)

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer, station))

    def get_steady_single_station(self):

        steady_list = []

        for station in range(self.num_stations):
            steady_list.append(self.num_cust_durations[station] / self.num_cust_durations[station].sum())

        return np.array(steady_list).reshape(self.num_stations, 500)





def get_ph():
    if sys.platform == 'linux':
        path = '/scratch/eliransc/ph_samples'
    else:
        path = r'C:\Users\Eshel\workspace\data\PH_samples'

    folders = os.listdir(path)

    folder_ind =   np.random.randint(len(folders))
    files = os.listdir(os.path.join(path, folders[folder_ind]))
    ind_file1 = np.random.randint(len(files))

    data_all = pkl.load(open(os.path.join(path,folders[folder_ind], files[ind_file1]), 'rb'))

    return data_all
rhos_list = []
for sample in tqdm(range(500)):

    try:
        begin = time.time()

        arrivals = get_ph()
        rho = np.random.uniform(0.02, 0.95)
        arrive_moms = []
        for mom in range(1, 11):
            arrive_moms.append(factorial(mom) / 1 ** mom)

        moms_arrive = arrivals[2]


        services_1 = get_ph()
        services_2 = get_ph()

        num_servers = 2

        sum_rates = 1 / np.random.uniform(1 / 20, 1 / 1.035)

        rate_1 = np.random.uniform(0.03, sum_rates)
        rate_2 = sum_rates - rate_1

        mean_ser_1 = 1/rate_1
        mean_ser_2 = 1 / rate_2

        services_norm_1 =  services_1[3] / rate_1

        A = services_1[1] * rate_1
        a = services_1[0]

        moms_ser_1 = np.array(compute_first_n_moments(a, A, 10)).flatten()

        mom_1_ser_1 = moms_ser_1[0]
        mom_2_ser_1 = moms_ser_1[1]

        var_ser_1 = mom_2_ser_1 - mom_1_ser_1 ** 2
        scv_ser_1 = var_ser_1 / mom_1_ser_1 ** 2


        services_norm_2 = services_2[3] / rate_2

        A = services_2[1] * rate_2
        a = services_2[0]


        moms_ser_2 = np.array(compute_first_n_moments(a, A, 10)).flatten()

        mom_1_ser_2 = moms_ser_2[0]
        mom_2_ser_2 = moms_ser_2[1]

        var_ser_2 = mom_2_ser_2 - mom_1_ser_2 ** 2
        scv_ser_2 = var_ser_2 / mom_1_ser_2 ** 2


        scv_ser = max(scv_ser_1, scv_ser_2)

        if rho > 0.8:
            rho_factor = 1.25
        elif rho > 0.6:
            rho_factor = 1.1
        elif rho > 0.4:
            rho_factor = 1.05
        else:
            rho_factor = 1.

        if scv_ser > 10:
            scv_ser_factor = 1.25
        elif scv_ser > 4:
            scv_ser_factor = 1.15
        elif scv_ser > 2:
            scv_ser_factor = 1.05
        else:
            scv_ser_factor = 1.


        sim_time = 60000000
        sim_time = int(sim_time * rho_factor * scv_ser_factor)
        mu = 1.0
        num_stations = 1

        # print(num_servers)

        lamda = rate_1
        inps = []
        outputs1 = []
        outputs2 = []

        for trails in range(1):

            n_Queue_single_station = N_Queue_single_station(sim_time, num_stations, [services_norm_1, services_norm_2],  arrivals[3],
                                                            num_servers)
            n_Queue_single_station.run()

            input_ = np.concatenate((moms_arrive, moms_ser_1, moms_ser_2), axis=0)
            # output = n_Queue_single_station.get_steady_single_station()

            end = time.time()

            # print(end - begin)

            model_num = np.random.randint(1, 10000000)

            ########### output ############

            station = 0

            ####### Input ################

            inp_steady_0 = np.concatenate((np.log(moms_arrive), np.log(moms_ser_1), np.log(moms_ser_2), np.array([num_servers])))
            inps.append(inp_steady_0)
            ###############################
            ########### output ############

            output1 = n_Queue_single_station.get_steady_single_station()[0]
            # output2 = np.array(n_Queue_single_station.sojourn_times).mean().item()

            mean_val = (np.arange(output1.shape[0])*output1).sum()

            outputs1.append(output1)
            # outputs2.append(output2)
            print(n_Queue_single_station.busy_times[0]/sim_time, n_Queue_single_station.busy_times[1]/sim_time)
            # print(1-outputs1[0][0],outputs1[0][1],  rho, mean_ser_1, mean_ser_2, 1/(1/mean_ser_2+1/mean_ser_1))
            # rhos_list.append([n_Queue_single_station.busy_times[0]/sim_time, n_Queue_single_station.busy_times[1]/sim_time])


        if sys.platform == 'linux':
            path_steady_0 = '/scratch/eliransc/2_servers_hetro'
        else:
            path_steady_0 = r'C:\Users\Eshel\workspace\data\ggc_training_data'

        file_name =  'rho_' + str(rho)[:5] + '_num_servers_' + str(num_servers) + '_sim_time_' + str(sim_time) + 'steady_' + str(
            model_num) + '.pkl'

        full_path_steady_0 = os.path.join(path_steady_0, file_name)
        pkl.dump((inps, outputs1), open(full_path_steady_0, 'wb'))
        pkl.dump(rhos_list, open(r'C:\Users\Eshel\workspace\data\mom_mathcher_data/rho_list.pkl', 'wb'))
    except:
        print('exceed 500')

