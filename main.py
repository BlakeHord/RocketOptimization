import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import G
import scipy.integrate as integrate
from matplotlib.patches import Circle
from rocketcea.cea_obj import CEA_Obj
import random
from deap import base, creator, tools
import time


start = time.time()

toggle_optimize = False
plot_single = False
plot_spacex = True
optimize_single = False

#constants
m_over_ft = 1/3.2808399
C_to_K = 273.15
F_to_R = 459.67
K_over_R = 5/9
Pa_over_psi = 6895
kg_m3_over_slug_cuft = 515.379
g = 9.81 #m/s^2
earth_r = 6371000 #m
earth_m = 5.9722e24 #kg
pi = np.pi
payload = 1000 #kg

altitude_target = 300e3 #m
ox = "LOX"
fuel = "RP1"
C_obj = CEA_Obj(oxName=ox, fuelName=fuel)


boundary_list = [[2,3],[100,4500],[2,400],[20e3,8000e3],[1,45],[60,250],[0,30],[35,180], \
    [2,3],[100,4500],[2,400],[20e3,8000e3],[1,1],[100,500],[25000,25000],[35,180],[0,100], \
    [10,300],[1,10],[1e3,1e3]]
boundary_low = []
boundary_high = []
for bound in boundary_list:
    boundary_low.append(bound[0])
    boundary_high.append(bound[1])

#examples of current vehicles
#[payload mass, total mass, $price]
rocketlab_electron = [300, 12500]#, 7e6] #2 stage liquid
abl_rs1 = [1350, 0]#, 12e6] #2 stage liquid
virgin_launcherone = [500,27215]#,12e6] #2 stage liquid (air-launched)
russia_strela = [1400,105000]#,10.5e6] #only to 200km, 2 stage liquid
russia_rokot = [1950,107000]#,36e6] #only to 200km, 3 stage liquid
astra_32 = [150,0]#,2.5e6] #2 stage liquid
firefly_alpha = [1000,54000]#,15e6] #2 stage liquid

orbital_minotaur = [1458,73000]#,40e6] #4 stage solid
arianespace_vega = [1450,137000]#,25e6] #4 stage, 3 solid, 1 liquid
northrop_pegasus = [443,18500]#,40e6] #3 stage solid (air-launched)
jaxa_epsilon = [700,86182]#,39e6] #3 stage solid


# model of atmosphere to output temperature, pressure, density
def atm_model(altitude): #input in m
    altitude_ft = altitude / m_over_ft #ft
    h1 = 36152 #ft - upper limit of troposphere, start of stratosphere
    h2 = 82345 #ft - border between lower and upper stratosphere

    if (altitude_ft <= h1):
        T = (59 - 0.00356 * altitude) #F
        P = 2116 * ((T + F_to_R)/518.6)**5.256 #lbs/ft2
    elif (altitude_ft > h1 and altitude_ft < h2):
        T = -70 #F
        P = 473 * np.exp(1.73 - 0.000048 * altitude_ft) #lbs/ft2
    else:
        T = -205.05 + 0.00164*altitude_ft #F
        P = 51.97 * ((T+F_to_R)/389.98)**(-11.388) #lbs/ft2

    rho = P/(1718*(T+F_to_R)) #slugs/cu ft

    T = (T + F_to_R) * K_over_R #K
    P = P/144 * Pa_over_psi #Pa
    rho = rho * kg_m3_over_slug_cuft #kg/m^3

    return [T,P,rho]    

# returns force of drag (N) from altitude (m) and speed (m/s)
def drag_force(altitude, speed, d):
    [T,P,rho_a] = atm_model(altitude)
    frontal_area = pi*d*d/4
    Cd = 0.25 #from here: http://www.aerospaceweb.org/question/aerodynamics/q0231.shtml
    return (- 0.5 * Cd * frontal_area * rho_a * speed**2)

# equation for thrust angle dependent on altitude
def get_thrust_angle(alt, alpha1, alpha2):
    if alt < alpha1:
        angle = 0
    elif alt < (alpha1 + alpha2):
        angle = (1 - np.cos(pi*(alt - alpha1)/alpha2))*pi/4
    else:
        angle = pi/2
    return angle

# ODEs integrated in trajectory_sim of rocket's motion
def rocket_ode(t, x, stage, grav, acc_array, angle_arr):
    vy = x[0] #m/s - velocity in y
    vx = x[1] #m/s - velocity in x
    y_pos = x[2] #m - current y position
    x_pos = x[3] #m - current x position
    m_tot = x[4] #kg - current total mass

    v_tot = np.hypot(vx,vy) #magnitude of resultant velocity

    R = np.hypot(x_pos,y_pos) #m - from center of earth
    angle_loc = np.arctan2(y_pos,x_pos) #radians
    #print(angle_loc)
    alt = R - earth_r #m - from surface of earth

    theta_tilt = get_thrust_angle(alt,stage.alpha[0]*1000,stage.alpha[1]*1000) #radians - angle below radial of thrust
    angle_arr.append(angle_loc)

    drag = drag_force(alt,v_tot,stage.d)
    thrust_y = stage.F*np.sin(angle_loc - theta_tilt)#N
    thrust_x = stage.F*np.cos(angle_loc - theta_tilt)#N

    a_gravity = -G * earth_m / R**2
    a_gravity_y = a_gravity * np.sin(angle_loc)
    a_gravity_x = a_gravity * np.cos(angle_loc)

    grav.append([a_gravity_y, a_gravity_x])

    dx = np.zeros(5)

    if (R < earth_r): #if rocket has hit the ground
        dx[0] = 0
        dx[1] = 0
        dx[2] = 0
        dx[3] = 0
        dx[4] = 0
    elif (m_tot <= stage.m_dry): #if rocket has expended all fuel
        dx[0] = (drag*np.sin(angle_loc))/ m_tot + a_gravity_y #change in velocity y 
        dx[1] = (drag*np.cos(angle_loc))/ m_tot + a_gravity_x #change in velocity x
        dx[2] = vy #change in position y
        dx[3] = vx #change in position x
        dx[4] = 0 #change in mass
    else: #if rocket is still burning fuel
        dx[0] = ((thrust_y + drag*np.sin(angle_loc)) / m_tot + a_gravity_y)
        dx[1] = ((thrust_x + drag*np.cos(angle_loc)) / m_tot + a_gravity_x)
        dx[2] = vy
        dx[3] = vx
        dx[4] = -stage.mdot

    acc_array.append([dx[0],dx[1]])

    return dx

# simulates trajectory of stage given initial conditions
def trajectory_sim(stage, i_cond = None, plt = True):
    time_range = [0, stage.t_burn+stage.t_coast]
    if i_cond == None:
        initial_conditions = [0, 0, 0, earth_r + 10, stage.m_tot]
    else:
        initial_conditions = i_cond
    grav = []
    thrust = []
    angle = []

    extra_parameters = [stage, grav, thrust, angle]

    sol = integrate.solve_ivp(rocket_ode, time_range, initial_conditions, args=extra_parameters, max_step=100)

    if plt:
        fig1, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7, ax8, ax9))  = plt.subplots(3,3, figsize=(8,8))

        ax1.plot(sol.y[3],sol.y[2])
        ax1.set_xlim([earth_r*-1.1,earth_r*1.1])
        ax1.set_ylim([-earth_r*0.5,earth_r*0.5])
        earth_circle = Circle((0,0), earth_r, facecolor=(0.9,0.9,0.9))
        ax1.add_patch(earth_circle)
        ax1.set_title("Position")
        ax1.axis('equal')

        ax2.plot(sol.t, np.hypot(sol.y[3],sol.y[2])-earth_r)
        ax2.set_title("Altitude")

        ax3.plot(sol.t, np.hypot(sol.y[0], sol.y[1]))
        ax3.set_title("Velocity")

        grav_x = []
        grav_y = []
        for g in grav:
            grav_x.append(g[1])
            grav_y.append(g[0])

        ax4.plot(np.hypot(grav_x,grav_y))
        ax4.set_title("gravity")

        t_x = []
        t_y = []
        for t in thrust:
            t_y.append(t[0])
            t_x.append(t[1])

        ax5.plot(t_y)
        ax5.set_title("Acceleration Y")
        ax6.plot(t_x)
        ax6.set_title("Acceleration X")

        ax7.plot(angle)
        ax7.set_title("angle")

        ax8.plot(sol.y[0])
        ax9.plot(sol.y[1])
        ax8.set_title("Velocity Y")
        ax9.set_title("Velocity X")

        #fig1.tight_layout()
        plt.show()

    trajectory = [sol, np.hypot(sol.y[3],sol.y[2])]

    return trajectory


class rocket: 
    def __init__(self,stages):
        self.stages = stages
        self.trajectory = []
        self.max_altitude = 0 #m
        self.time = [] #s
        self.altitude = [] #m
        self.perigee = 0 #m
        self.alt_equiv = 0 #m
        self.glow = stages[0].m_tot #kg
        self.m_payload = stages[-1].m_payload #kg

    def add_to_altitude(self,altitude):
        for a in altitude:
            self.altitude.append(a - earth_r)

    def add_to_time(self,time):
        if len(self.time) == 0:
            last = 0
        else:
            last = self.time[-1]
        for t in time:
            self.time.append(t+last)

    def get_perigee(self):
        if self.altitude[-1] > 1e3:
            min_index = 0
            total_burn_time = self.stages[0].t_burn + self.stages[0].t_coast + self.stages[1].t_burn
            for i in range(len(self.time)):
                if self.time[i] > total_burn_time:
                    min_index = i
                    break

            self.perigee = np.min(self.altitude[min_index:])

            if (self.perigee < 50e3):
                self.alt_equiv = 0
            else:
                self.alt_equiv = (self.perigee + self.max_altitude)/2
        else:
            self.alt_equiv = 0

    def simulate(self):
        i_cond = None
        self.trajectory = []
        for s in self.stages:
            if i_cond != None:
                i_cond[4] = s.m_tot
            trajectory_i = trajectory_sim(s,i_cond,False)
            s.trajectory = trajectory_i
            self.trajectory.append(s.trajectory[0])
            self.add_to_altitude(s.trajectory[1])
            self.add_to_time(s.trajectory[0].t)
            i_cond = [trajectory_i[0].y[0,-1], trajectory_i[0].y[1,-1], trajectory_i[0].y[2,-1], trajectory_i[0].y[3,-1], 0]
        self.max_altitude = np.max(self.altitude)
        self.get_perigee()

    def plot(self, traj=True, alt=True, show=True, fig = None, axes = None):
        if self.max_altitude == 0:
            self.simulate()

        if traj == False and alt == False:
            return 0

        if fig == None:
            if traj and alt:
                fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,5))
                axes = [ax1,ax2]
            else:
                fig, ax1 = plt.subplots(1,1, figsize = (5,5))
                axes = [ax1]

        if traj:
            for s in self.stages:
                axes[0].plot(s.trajectory[0].y[3],s.trajectory[0].y[2])
            earth_circle = Circle((0,0), earth_r, facecolor=(0.9,0.9,0.9))
            axes[0].add_patch(earth_circle)
            circle_target = Circle((0,0), earth_r + altitude_target, fill=False)
            axes[0].add_patch(circle_target)

            axes[0].set_title("Position")
            axes[0].axis('equal')

            if alt:
                axes[1].plot(self.time, self.altitude)
                axes[1].set_title("Altitude vs Time")
                axes[1].ticklabel_format(axis="Y", style="sci", scilimits=(3,3))
        else: 
            axes[0].plot(self.time, self.altitude)
            axes[0].set_title("Altitude vs Time")
            axes[0].ticklabel_format(axis="Y", style="sci", scilimits=(3,3))
        
        if show:
            plt.show()

        return fig, axes

class stage:
    def __init__(self,MR,pc,area_ratio,F,n_engine,t_burn,t_coast,thrust_to_weight,d,alpha,m_payload,vac,mass_factor=1):
        ox_rho = 1141 #kg/m3 - from https://en.wikipedia.org/wiki/Liquid_oxygen
        fuel_rho = 810 #kg/m3 - from http://webserver.dmt.upm.es/~isidoro/bk3/c15/Fuel%20properties.pdf

        if vac == False:
            [Isp, mode] = C_obj.estimate_Ambient_Isp(Pc=pc, MR=MR, eps=area_ratio)
            p_o = atm_model(0)[1] / Pa_over_psi #psi - pressure at sea level
            Cf = C_obj.get_PambCf(Pamb=p_o, Pc=pc, MR=MR, eps=area_ratio)[1]
            self.pc_Pa = pc * Pa_over_psi #Pa
            self.A_star = F/(Cf*self.pc_Pa) #m^2
        else:
            Isp = C_obj.get_Isp(Pc=pc, MR=MR, eps=area_ratio)
            p_o = atm_model(0)[1] / Pa_over_psi #psi - pressure at sea level
            Cf = C_obj.get_PambCf(Pamb=p_o, Pc=pc, MR=MR, eps=area_ratio)[1]
            self.pc_Pa = pc * Pa_over_psi #Pa
            self.A_star = F/(Cf*self.pc_Pa) #m^2

        self.MR = MR
        self.area_ratio = area_ratio
        self.pc = pc
        C_amb = Isp * g #m/s
        #print(Isp)
        self.mdot = F*n_engine/C_amb #kg/s
        self.m_prop = self.mdot*t_burn #kg
        if self.m_prop < 0: 
            self.m_prop = 1

        ox_mass = self.m_prop*MR/(MR+1) #kg
        fuel_mass = self.m_prop/(MR+1) #kg
        tank_t = 5e-3*2.5 #5mm - 150% increase to match falcon 9 dry mass (accounted for by baffles, other structural elements)
        tank_rho = 2700 #kg/m3 - density of aluminum
        tank_volume = (ox_mass / ox_rho + fuel_mass / fuel_rho)*1.05 #m3 - plus 5% ullage pressurization volume
        tank_height = tank_volume/((d - 2*tank_t)**2/4*pi) #m
        tank_mass = d*pi*tank_t*tank_height*tank_rho #kg 

        #other masses: - from https://spacecraft.ssl.umd.edu/academics/791S16/791S16L08.MERsx.pdf
        insulation_mass = tank_height*d*pi*1.123 #kg
        wiring_mass = 1.058 * (self.m_prop)**0.5 * (tank_height)**0.25 #kg
        avionics_mass = 10*(self.m_prop)**0.361 #kg

        engine_mass = F/(thrust_to_weight*g) #kg
        engine_total_mass = engine_mass*n_engine #kg

        self.m_dry = mass_factor*(engine_total_mass+tank_mass+insulation_mass+wiring_mass+avionics_mass)+m_payload
        self.m_tot = self.m_dry + self.m_prop
        self.F = F*n_engine
        self.n_engine = n_engine
        self.alpha = alpha
        self.t_burn = t_burn
        self.t_coast = t_coast
        self.d = d
        self.alpha=alpha
        self.m_payload = m_payload
        self.trajectory = None

    def print_stage(self):
        print("M_Tot = ", self.m_tot, "kg\nM_Dry = ", self.m_dry, "kg\nF = ", self.F, "N")
        print("Mdot = ", self.mdot, "kg/s\nn_engine = ", self.n_engine, "\nd = ", self.d, "m")
        print("alpha = ", self.alpha, "km")
        print("Burn time = ", self.t_burn, "s\n\n")


#optimization functions
def checkvalid(individual):
    boundary_list = [[2,3],[100,4500],[2,400],[20e3,8000e3],[1,45],[60,250],[0,30],[35,180], \
    [2,3],[100,4500],[2,400],[20e3,8000e3],[1,1],[100,500],[25000,25000],[35,180],[0,100], \
    [10,300],[0.5,10],[payload,payload]]
    for i in range(len(individual)):
        if individual[i] < boundary_list[i][0]:
            individual[i] = boundary_list[i][0]
        elif individual[i] > boundary_list[i][1]:
            individual[i] = boundary_list[i][1]
    individual[4] = round(individual[4])

def evaluate(individual):
    i = individual[0]
    ox = "LOX"
    fuel = "RP1"
    checkvalid(i)
    alpha = [i[16],i[17]]
    d = i[18]
    m_payload = i[19]
    stage2 = stage(i[8],i[9],i[10],i[11],i[12],i[13],i[14],i[15],d,alpha,m_payload,True,0.7)
    stage1 = stage(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],d,alpha,stage2.m_tot,False,1.0)
    rocket1 = rocket([stage1,stage2])
    rocket1.simulate()
    return [rocket1.glow / rocket1.m_payload + max(0,altitude_target - rocket1.alt_equiv)]

def plot_evaluate(individual, min_fits=None):
    i = individual[0]
    ox = "LOX"
    fuel = "RP1"
    if min_fits != None:
        checkvalid(i)
    alpha = [i[16],i[17]]
    d = i[18]
    m_payload = i[19]
    stage2 = stage(i[8],i[9],i[10],i[11],i[12],i[13],i[14],i[15],d,alpha,m_payload,True,0.7)
    stage1 = stage(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],d,alpha,stage2.m_tot,False,1.0)
    rocket1 = rocket([stage1,stage2])
    rocket1.simulate()
    fig, axes = rocket1.plot(show=False)
    if min_fits != None:
        fig2, ax3 = plt.subplots(1,1, figsize=(4,4))
        ax3.plot(min_fits)
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Minimum Fitness (GLOW/payload mass")
    plt.show()

def random_init():
    i = [0]*20
    i[0] = random.uniform(2,3) #MR_1
    i[1] = random.uniform(100,4500) #pc_1
    i[2] = random.uniform(2,400) #area_ratio_1
    i[3] = random.uniform(20e3,8000e3) #F_1
    i[4] = random.randint(1,45) #n_engine_1
    i[5] = random.uniform(60,250) #t_burn_1
    i[6] = random.uniform(0,30) #t_coast_1
    i[7] = random.uniform(35,180) #thrust_to_weight_1
    i[8] = random.uniform(2,3) #MR_2
    i[9] = random.uniform(100,4500) #pc_2
    i[10] = random.uniform(2,400) #area_ratio_2
    i[11] = random.uniform(20e3,8000e3) #F_2
    i[12] = 1 #n_engine_2
    i[13] = random.uniform(200,500) #t_burn_2
    i[14] = 25000 #t_coast_2
    i[15] = random.uniform(35,180) #thrust_to_weight_2
    i[16] = random.uniform(0,100) #angle_thrust_1
    i[17] = random.uniform(10,300) #angle_thrust_2
    i[18] = random.uniform(0.5,10) #d
    i[19] = payload#random.uniform(1,2000) #m_payload
    return i

def optimize():
    pop = toolbox.population(n=100)
    print("Made population!")
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    print("Evaluated first population!")

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.5
    
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    min_fits = []
    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while g < 40:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values
            
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        min_fits.append(min(fits))

        print(min(fits), max(fits), mean, std)
    
    best = pop[np.argmin([toolbox.evaluate(x) for x in pop])]
    return best, min_fits

def print_individual(individual):
    i = individual[0]
    print("MR_1 = ", i[0])
    print("pc_1 = ", i[1])
    print("area_ratio_1 = ", i[2])
    print("F_1 = ", i[3])
    print("n_engine_1 = ", i[4])
    print("t_burn_1 = ", i[5])
    print("t_coast_1 = ", i[6])
    print("thrust_to_weight_1 = ", i[7])
    print("MR_2 = ", i[8])
    print("pc_2 = ", i[9])
    print("area_ratio_2 = ", i[10])
    print("F_2 = ", i[11])
    print("n_engine_2 = ", i[12])
    print("t_burn_2 = ", i[13])
    print("t_coast_2 = ", i[14])
    print("thrust_to_weight_2 = ", i[15])
    print("angle_thrust_1 = ", i[16])
    print("angle_thrust_2 = ", i[17])
    print("d = ", i[18])
    print("m_payload = ", i[19])
    print()

def make_rocket(individual):
    i = individual[0]
    alpha = [i[16],i[17]]
    d = i[18]
    m_payload = i[19]
    stage2 = stage(i[8],i[9],i[10],i[11],i[12],i[13],i[14],i[15],d,alpha,m_payload,True,0.7)
    stage1 = stage(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],d,alpha,stage2.m_tot,False,1.0)
    rocket1 = rocket([stage1,stage2])
    rocket1.simulate()
    return rocket1



if plot_spacex:
    #Design Variables - SpaceX
    ############ STAGE 1 ENGINE (8 parameters)
    MR_1 = 2.5 #Mixture Ratio (O/F Ratio)
    #range: 2-3 
    pc_1 = 1500 #chamber pressure - psi
    #range: 100-4500 (raptor ~4500)
    area_ratio_1 = 21 #nozzle area ratio (exit/throat)
    #range: 1-400
    F_1 = 7607e3/9 #sea level thrust - N
    #range: 20e3-8000e3
    t_burn_1 = 162 #burning time - s
    #range: 60-250
    t_coast_1 = 0 #coast time - s
    #range: 0-30
    n_engine_1 = 9 #number of engines on first stage (assume 1 on second stage)
    #range: 1-45 (shouldn't have a big impact)
    thrust_to_weight_1 = 180 #thrust to weight ratio of engine
    #range: 35-180 (higher is definitely better)

    ################## STAGE 2 ENGINE (8 parameters)
    MR_2 = 2.5 #Mixture Ratio (O/F Ratio)
    #range: 2-3 
    pc_2 = 1500 #chamber pressure - psi
    #range: 100-4500 (raptor ~4500)
    area_ratio_2 = 100 #nozzle area ratio (exit/throat)
    #range: 1-400
    F_2 = 934e3 #vacuum thrust - N
    #range: 20e3-8000e3
    t_burn_2 = 397 #burning time - s
    #range: 60-250
    t_coast_2 = 25000 #coast time - s
    #range: 0-30
    n_engine_2 = 1 #number of engines on first stage (assume 1 on second stage)
    #range: 1
    thrust_to_weight_2 = 180 #thrust to weight ratio of engine
    #range: 35-180 (higher is definitely better)

    ################### ROCKET (3 parameters)
    angle_thrust_1 = 25 #altitude to start angling rocket - km
    #range: 0-100
    angle_thrust_2 = 200 #additional altitude to complete the 90 degree turn (Eq 3 Bairstow) - km
    #range: 10-300
    alpha = [angle_thrust_1,angle_thrust_2]
    d = 3.7 #diameter - m
    #range: 0.5-10 (starship is 9m)
    m_payload = 15000 #kg

    stage2 = stage(MR_2,pc_2,area_ratio_2,F_2,n_engine_2,t_burn_2,t_coast_2,thrust_to_weight_2,d,alpha,m_payload,True,0.7)
    stage1 = stage(MR_1,pc_1,area_ratio_1,F_1,n_engine_1,t_burn_1,t_coast_1,thrust_to_weight_1,d,alpha,stage2.m_tot,False)
    rocket1 = rocket([stage1,stage2])
    rocket1.simulate()
    fig, axes = rocket1.plot(show=True)

if toggle_optimize:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("random_init", random_init)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.random_init, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=boundary_low, up=boundary_high, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=5)

    n_payloads = 75
    max_payload = 10000
    min_payload = 100
    payload_array = np.linspace(min_payload,max_payload,n_payloads)

    best_array_1 = []

    for count, payload in enumerate(payload_array):
        print(count)
        best, min_fits = optimize()
        print(best)
        best_rocket = make_rocket(best)
        best_array_1.append(best_rocket.glow)
        print(best_rocket.glow)

    fig, ax1 = plt.subplots(1,1, figsize=(8,5))

    ax1.plot(payload_array,best_array_1, 'k.')

    ax1.set_xlabel("Payload Mass to 300km (kg)")
    ax1.set_ylabel("Gross Liftoff Mass (kg)")

    X = np.arange(1, 12000, 100)
    Y = np.arange(1, 300000, 100)
    X, Y = np.meshgrid(X, Y)
    Z = (X/Y)
    levels = [0.03,0.04,0.05,0.06,0.07]
    CS = ax1.contour(X,Y,Z,levels=levels, linestyles='dashed', alpha=0.5)
    ax1.clabel(CS, inline=1, fontsize=10, fmt='%1.2f')

    ax1.plot(rocketlab_electron[0],rocketlab_electron[1], 'r.')
    ax1.annotate("1",rocketlab_electron)
    ax1.plot(virgin_launcherone[0],virgin_launcherone[1], 'r.')
    ax1.annotate("2",virgin_launcherone)
    ax1.plot(russia_strela[0],russia_strela[1], 'r.')
    ax1.annotate("3",russia_strela)
    ax1.plot(russia_rokot[0],russia_rokot[1], 'r.')
    ax1.annotate("4",russia_rokot)
    ax1.plot(firefly_alpha[0],firefly_alpha[1], 'r.')
    ax1.annotate("5",firefly_alpha)
    ax1.plot(orbital_minotaur[0],orbital_minotaur[1], 'r.')
    ax1.annotate("6",orbital_minotaur)
    ax1.plot(arianespace_vega[0],arianespace_vega[1], 'r.')
    ax1.annotate("7",arianespace_vega)
    ax1.plot(northrop_pegasus[0],northrop_pegasus[1], 'r.')
    ax1.annotate("8",northrop_pegasus)
    ax1.plot(jaxa_epsilon[0],jaxa_epsilon[1], 'r.')
    ax1.annotate("9",jaxa_epsilon)

    end = time.time()
    print("Time elapsed = ", round(end-start), "s\n")
    plt.show()

if plot_single:
    #copy and paste individual array from code output
    individual_2 = [[3, 3671, 31.312028697373634, 478242, 8, 65.93114274844316, 17, 103, 3, 962, 263, 2562492.543371337, 1, 203, 25000, 166, 25, 74, 9.882978563432367, 10792.0]]
    individual_1 = [[3, 4188, 20, 693185, 2, 64, 27, 164, 3, 4444, 386, 489594, 1, 169, 25000, 146.32431116786617, 41, 80.72528191389162, 7, 1667.5]]
    plot_evaluate(individual_1)

if optimize_single:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("random_init", random_init)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.random_init, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=boundary_low, up=boundary_high, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=5)

    paylaod = 10000 #kg
    best, min_fits = optimize()
    best_rocket = make_rocket(best)
    print(best_rocket.glow)
    plot_evaluate(best, min_fits)

