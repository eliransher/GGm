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
from scipy.special import gamma, factorial
from scipy.linalg import expm, sinm, cosm
from sympy import *
is_print = False

sys.path.append(r'C:\Users\Eshel\workspace\butools2\Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')

from butools.ph import *
from butools.map import *
from butools.queues import *
import time
from butools.mam import *
from butools.dph import *

arrival_dics = {0: 'LN025', 1: 'LN4', 2: 'G4', 3: 'E4', 4: 'H2'}
ser_dics = {0: 'LN025', 1: 'LN4', 2: 'G4', 3: 'E4', 4: 'H2', 5: 'M'}

def gamma_pdf(x, theta, k):
    return (1 / (gamma(k))) * (1 / theta ** k) * (np.exp(-x / theta))


def gamma_lst(s, theta, k):
    return (1 + theta * s) ** (-k)

def gamma_mfg(shape, scale, s):
    return (1-scale*s)**(-shape)

def get_nth_moment(shape, scale, n):
    s = Symbol('s')
    y = gamma_mfg(shape, scale, s)
    for i in range(n):
        if i == 0:
            dx = diff(y,s)
        else:
            dx = diff(dx,s)
    return dx.subs(s, 0)


def gamma_lst(s, theta, k):
    return (1 + theta * s) ** (-k)


def unif_lst(s, b, a=0):
    return (1 / (b - a)) * ((np.exp(-a * s) - np.exp(-b * s)) / s)


def n_mom_uniform(n, b, a=0):
    return (1 / ((n + 1) * (b - a))) * (b ** (n + 1) - a ** (n + 1))


def laplace_mgf(t, mu, b):
    return exp(mu * t) / (1 - (b ** 2) * (t ** 2))


def nthmomlap(mu, b, n):
    t = Symbol('t')
    y = laplace_mgf(t, mu, b)
    for i in range(n):
        if i == 0:
            dx = diff(y, t)
        else:
            dx = diff(dx, t)
    return dx.subs(t, 0)


def normal_mgf(t, mu, sig):
    return exp(mu * t + (sig ** 2) * (t ** 2) / 2)


def nthmomnormal(mu, sig, n):
    t = Symbol('t')
    y = normal_mgf(t, mu, sig)
    for i in range(n):
        if i == 0:
            dx = diff(y, t)
        else:
            dx = diff(dx, t)
    return dx.subs(t, 0)


def generate_unif(is_arrival):
    if is_arrival:
        b_arrive = np.random.uniform(2, 4)
        a_arrive = 0
        moms_arr = []
        for n in range(1, 11):
            moms_arr.append(n_mom_uniform(n, b_arrive))
        return (a_arrive, b_arrive, moms_arr)
    else:
        b_ser = 2
        a_ser = 0
        moms_ser = []
        for n in range(1, 11):
            moms_ser.append(n_mom_uniform(n, b_ser))
        return (a_ser, b_ser, moms_ser)


def get_hyper_ph_representation(mu):
    p1 = 0.5 + (-2. + 0.5 * mu) / (16. + -mu ** 2) ** 0.5
    p2 = 1 - p1
    lam1 = 1 / (2 + 0.5 * mu + 0.5 * (16 - mu ** 2) ** 0.5)
    lam2 = 1 / (2 + 0.5 * mu - 0.5 * (16 - mu ** 2) ** 0.5)

    s = np.array([[p1, p2]])
    A = np.array([[-lam1, 0], [0, -lam2]])
    return (s, A)

def log_normal_gener(mu, sig2, sample_size):
    m = np.log((mu**2)/(sig2+mu**2)**0.5)
    v = (np.log(sig2/mu**2+1))**0.5
    s = np.random.lognormal(m, v, sample_size)
    return s

def compute_first_ten_moms_log_N(s):
    moms = []
    for ind in range(1,11):
        moms.append((s**ind).mean())
    return moms

def generate_gamma(is_arrival, rho = 0.01):
    if is_arrival:
        # rho = np.random.uniform(0.7, 0.99)
        shape = 0.25/rho # 0.25 # np.random.uniform(0.1, 100)
        scale =  4 #1 / (rho * shape)
        moms_arr = np.array([])
        for mom in range(1, 11):
            moms_arr = np.append(moms_arr, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
        return (shape, scale, moms_arr)
    else:
        shape = 0.25 # np.random.uniform(1, 100)
        scale = 1 / shape
        moms_ser = np.array([])
        for mom in range(1, 11):
            moms_ser = np.append(moms_ser, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
        return (shape, scale, moms_ser)

def ser_moment_n(s, A, mom):
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) *factorial(mom)*np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False

def compute_first_n_moments(s, A, n=3):
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list




def create_Erlang4(lam):
    s = np.array([[1, 0, 0, 0]])

    A = np.array([[-lam, lam, 0, 0], [0, -lam, lam, 0], [0, 0, -lam, lam], [0, 0, 0, -lam]])

    return (s, A)


def generate_normal(is_arrival):
    if is_arrival:
        mu = np.random.uniform(1.1, 2.0)
        sig = np.random.uniform(mu / 6, mu / 4)

        moms_arr = np.array([])
        for mom in tqdm(range(1, 11)):
            moms_arr = np.append(moms_arr, np.array(N(nthmomnormal(mu, sig, mom))).astype(np.float64))

        return (mu, sig, moms_arr)
    else:
        mu = 1
        sig = np.random.uniform(0.15, 0.22)

        moms_ser = np.array([])
        for mom in tqdm(range(1, 11)):
            moms_ser = np.append(moms_ser, np.array(N(nthmomnormal(mu, sig, mom))).astype(np.float64))
        return (mu, sig, moms_ser)



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

def get_setup():
    rhos = np.linspace(0.01, 0.94, 18)
    num_servers = np.linspace(1, 10, 10).astype(int)

    rho = np.random.choice(rhos).item()
    num_server = 1 # np.random.choice(num_servers).item()
    ser_key_1 = np.random.choice(np.arange(len(ser_dics.keys()))).item()
    ser_key_2 = np.random.choice(np.arange(len(ser_dics.keys()))).item()
    arrive_key = np.random.choice(np.arange(len(arrival_dics.keys()))).item()

    return (rho, num_server, arrive_key, ser_key_1, ser_key_2)


def get_moms_realizations(rho, num_server, dist, sample_size, is_arrive):
    if dist == 'LN025':
        scv = 0.25
        times = log_normal_gener(1, scv, sample_size)
        if is_arrive == True:
            mean = 1
        else:
            mean = rho * num_server
        times = times * mean
        moms = compute_first_ten_moms_log_N(times)

    if dist == 'LN4':
        scv = 4.0
        times = log_normal_gener(1, scv, sample_size)
        if is_arrive == True:
            mean = 1
        else:
            mean = rho * num_server
        times = times * mean
        moms = compute_first_ten_moms_log_N(times)

    if dist == 'G4':

        shape, scale, moms_ = generate_gamma(False, rho)

        times = np.random.gamma(shape, scale, sample_size)
        if is_arrive == True:
            mean = 1
        else:
            mean = rho * num_server
        times = times * mean
        moms = compute_first_ten_moms_log_N(times)

    if dist == 'E4':
        # Parameters
        shape_k = 4  # Shape parameter (must be an integer)
        scale_theta = 0.25  # Scale parameter (θ)

        # Generate samples
        times = np.random.gamma(shape_k, scale_theta, size=sample_size)
        if is_arrive == True:
            mean = 1
        else:
            mean = rho * num_server
        times = times * mean
        moms = compute_first_ten_moms_log_N(times)

    if dist == 'H2':

        s, A = get_hyper_ph_representation(1)
        times = SamplesFromPH(ml.matrix(s), A, sample_size)
        if is_arrive == True:
            mean = 1
        else:
            mean = rho * num_server

        times = times * mean
        moms = compute_first_ten_moms_log_N(times)

    if dist == 'M':
        times = np.random.exponential(1, sample_size)
        if is_arrive == True:
            mean = 1
        else:
            mean = rho * num_server
        times = times * mean
        moms = compute_first_ten_moms_log_N(times)

    moms = [mom.item() for mom in moms]

    return moms, times



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

sample_size = 50000000

for sample in tqdm(range(500)):

    if True: #try

        begin = time.time()

        rho, num_server, arrive_key, ser_key_1, ser_key_2 = get_setup()
        sum_rates = 1/rho

        rate_1 = np.random.uniform(0.03, sum_rates)
        rate_2 = sum_rates - rate_1

        moms_arrive, arrive_times = get_moms_realizations(rho, num_server, arrival_dics[arrive_key], sample_size, True)
        moms_ser1, ser_times1 = get_moms_realizations(1/rate_1, num_server, ser_dics[ser_key_1], sample_size, False)
        moms_ser2, ser_times2 = get_moms_realizations(1/rate_2, num_server, ser_dics[ser_key_2], sample_size, False)
        moms_arrive = np.array(moms_arrive)
        moms_ser1 = np.array(moms_ser1)
        moms_ser2 = np.array(moms_ser2)

        mom_1_ser_1 = moms_ser1[0]
        mom_2_ser_1 = moms_ser1[1]

        mom_1_ser_2 = moms_ser2[0]
        mom_2_ser_2 = moms_ser2[1]

        var_ser1 = mom_2_ser_1 - mom_1_ser_1 ** 2
        scv_ser1 = var_ser1 / mom_1_ser_1 ** 2

        var_ser2 = mom_2_ser_2 - mom_1_ser_2 ** 2
        scv_ser2 = var_ser2 / mom_1_ser_2 ** 2

        ###############################################

        begin = time.time()

        num_servers = 2


        scv_ser = max(scv_ser1, scv_ser2)

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

            n_Queue_single_station = N_Queue_single_station(sim_time, num_stations, [ser_times1, ser_times2],  arrive_times,
                                                            num_servers)
            n_Queue_single_station.run()

            input_ = np.concatenate((moms_arrive, moms_ser1, moms_ser2), axis=0)
            # output = n_Queue_single_station.get_steady_single_station()

            end = time.time()

            # print(end - begin)

            model_num = np.random.randint(1, 10000000)

            ########### output ############

            station = 0

            ####### Input ################

            inp_steady_0 = np.concatenate((np.log(moms_arrive), np.log(moms_ser1), np.log(moms_ser2)))
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
            # rhos_list.append([n_Queue_single_station.busy_times[0]/sim_time, n_Queue_single_station.busy_times[1]/sim_time], rate_1, rate_2)
            pkl.dump(rhos_list, open(r'C:\Users\Eshel\workspace\data\mom_mathcher_data/rho_list.pkl', 'wb'))

        if sys.platform == 'linux':
                path_steady_0 = '/scratch/eliransc/2_servers_hetro_special'
        else:
            path_steady_0 = r'C:\Users\Eshel\workspace\data\hetro_data\ggc_test_data_special'

        file_name = ('rho_' + str(rho)[:5] + '_num_servers_' + str(num_server) + '_arrival_dist_' + arrival_dics[
            arrive_key]
                     + '_ser_dist1_' + ser_dics[ser_key_1]+ '_ser_dist2_' + ser_dics[ser_key_2] + '_sim_time_' + str(sim_time) + 'steady_' + str(
                    model_num) + '.pkl')

        full_path_steady_0 = os.path.join(path_steady_0, file_name)
        pkl.dump((inps, outputs1), open(full_path_steady_0, 'wb'))
        # pkl.dump(rhos_list, open(r'C:\Users\Eshel\workspace\data\mom_mathcher_data/rho_list.pkl', 'wb'))
    # except:
    #     print('exceed 500')

