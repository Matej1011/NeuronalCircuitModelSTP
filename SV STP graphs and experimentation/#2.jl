using Distributions, Random, Statistics, SparseArrays, Plots, JLD, InvertedIndices, NaNStatistics, LaTeXStrings, Base.Threads, CSV, DataFrames
#include("/home/ge74coy/mnt/naspersonal/Code/PhD/paper_code/functions.jl");
#    julia /home/ge74coy/mnt/naspersonal/Code/PhD/paper_code/training_both_final.jl
################################################################################

function save_param(x_name, x_tuple)
    x_name = replace(x_name, "\n" => "")
    x_name = replace(x_name, " " => "")
    x_name = split(x_name, ",")
    dict_param = Dict{String,Any}()
    for (j, i) in enumerate(x_tuple)
        for (n, m) in enumerate(x_name)
            if j == n
                dict_param[m] = i
            end
        end
    end
    return dict_param
end

function sim(zzz) #runs simulation given weight matrix and subpopulations
    T, N_trials, stim_rate, train_start, t_weightsave, time_weightsave,
    save_weights, save_spikes, save_weights_name, save_partial, save_all, save_weight_partial,
    je0, jp0, js0, jv0, r_ext, SST_hebb, PV_symmetric,
    Ne, Np, Ns, Nv, Ncells, Nspikes, Npop, Nmaxmembers, Ne_not,
    jee0, jpe, jse, jep0, jpp, jvp, jes0, jps, jsv, jvs, jve, percentage_dis_inh,
    p_EE, p_EI, p_IE, p_II, name, stim_time, gap_time, range_plast_sst,
    taue, taup, taus, tauv, vth_e, vth_i, vre, taurefrac, tauedecay, taupdecay, tausdecay, tauvdecay,
    jeemin, jeemax, jepmin, jepmax, jesmin, jesmax,
    tau_ipv, tau_isst, tau_plus, tau_minus, tau_x, tau_y, A2_minus, A2_plus, A3_minus, A3_plus,
    eta_pv, r0_pv, a_pre, a_post, dt, dtnormalize, used_param = parameters()

    weights = zeros(Float64, Ncells, Ncells)
    weights[1:Ne, (1+Ne):(Ne+Np)] .= (jpe) #E to PV
    weights[1:Ne, (1+Ne+Np):(Ne+Np+Ns)] .= (jse) #E to SST
    weights[1:Ne, (1+Ne+Np+Ns):Ncells] .= (jve) #E to VIP
    weights[(1+Ne):(Ne+Np), 1:Ne] .= (jep0) #PV to E start
    weights[(1+Ne):(Ne+Np), (1+Ne):(Ne+Np)] .= (jpp) #PV to PV
    weights[(1+Ne):(Ne+Np), (1+Ne+Np+Ns):Ncells] .= (jvp) #PV to VIP
    weights[(1+Ne+Np):(Ne+Np+Ns), 1:Ne] .= (jes0) #SST to E start
    weights[(1+Ne+Np):(Ne+Np+Ns), (1+Ne):(Ne+Np)] .= (1*jps) #SST to PV
    weights[(1+Ne+Np):(Ne+Np+Ns), (1+Ne+Np+Ns):Ncells] .= jvs #SST to VIP
    weights[(1+Ne+Np+Ns):Ncells, (1+Ne+Np):(Ne+Np+Ns)] .= (jsv) #VIP to SST
    weights[1:Ne, 1:Ne] .= 1*jee0 #E to E

    #parameter for printing out less stuff
    cc_prev = 0

    #in Hz
    frequency = 50

    time1 = time()

    randomconnections = zeros(Float64, Ncells, Ncells)   #generating random connections, easy way
    randomconnections[(1:Ne), (1:Ne)] = (rand(Ne, Ne) .< p_EE) # E to E
    randomconnections[(Ne+1):Ncells, (1:Ne)] = (rand(Ncells - Ne, Ne) .< p_EI) #I to E
    randomconnections[(1:Ne), (Ne+1):Ncells] = (rand(Ne, Ncells - Ne) .< p_IE) #E to I
    randomconnections[(Ne+1):Ncells, (Ne+1):Ncells] = (rand(Ncells - Ne, Ncells - Ne) .< p_II) #I to I

    weights = weights .* randomconnections
    for cc = 1:Ncells
        weights[cc, cc] = 0 #no autapses == self-loops
    end

    #

    ###initialisation###
    times = zeros(Ncells, Nspikes) #generate matrix with spike times for each neuron
    ns = zeros(Int64, Ncells) #spike recording for each neuron (accumulated number of spikes for each neuron)

    #recurrent connection currents
    Ie, Ip, Is, Iv = zeros(Float64, Ncells), zeros(Float64, Ncells), zeros(Float64, Ncells), zeros(Float64, Ncells)# Currents, V/s

    #baseline input currents (drawn from normal distributions)
    I_ext = zeros(Float64, Ncells)
##############################################
    alpha = 5.6
    C = 1.2

    I_ext[1:Ne] .= alpha
    I_ext[(1+Ne):(Ne+Np)] .= alpha
    I_ext[(1+Ne+Np+Ns):Ncells] .= C
##############################################
    rx = ones(Float64, Ncells) * r_ext #input rates

    v = zeros(Float64, Ncells) #membrane voltage
    sumwee0 = zeros(Float64, Ne) #array for initial summed E weight for every E neuron; for homeostatic normalisation
    Nee = zeros(Int64, Ne) #later number of incoming E->E inputs for every E neuron; for homeostatic normalisation

    weight_matrix_save = Dict()
    if save_all
        parameters_save = Dict()
    end


    #SETTING POTENTIALS RANDOMLY
    for cc = 1:Ncells #for every neuron
        if cc <= Ne #for excitatory neurons
            v[cc] = vre + (vth_e - vre) * rand() #random starting voltage between reset and threshold potential
            for dd = 1:Ne
                sumwee0[cc] += weights[dd, cc] #initial summed excitatory input to an E neuron
                if weights[dd, cc] > 0
                    Nee[cc] += 1 #number of E to E connections for subtractive normalisation later
                end
            end
        else
            v[cc] = vre + (vth_i - vre) * rand() #random starting voltage between reset and threshold potential
        end
    end

    lastSpike = zeros(Ncells) #will record last spike time for each neuron

    x = zeros(Float64, Ncells) #iSTDP, spike detector
    y = zeros(Float64, Ncells) #iSTDP, spike detector
    r1 = zeros(Float64, Ncells) #tSTDP, presynaptic detector
    r2 = zeros(Float64, Ncells) #tSTDP, presynaptic detector
    o1 = zeros(Float64, Ncells) #tSTDP, postsynaptic detector
    o2 = zeros(Float64, Ncells) #tSTDP, postsynaptic detector

    Nsteps = round(Int, T / dt) #timesteps for simulation
    
    inormalize = round(Int, dtnormalize / dt) #homeostatic normalisation every inormalised step

    println("starting simulation")

    ######BEGIN MAIN SIMULATION LOOP######
    range_of_saving = collect(round(Int, train_start):T/10:T)  #save 1/10 of values 
    count_add = 0
    ns_prev = zeros(Int64, Ncells)
    tt_prev = 0

    depression_spike = zeros(Ncells)
    facilitation_spike = zeros(Ncells)

    Counting = 0
    for tt = 1:Nsteps #loop over time
        ###PROGRESS###
        if mod(tt, Nsteps / 100) == 1  #print percent complete
            println("\r", round(Int, 100 * tt / Nsteps))

            println("mean excitatory synaptic weight: ", mean(weights[1:Ne, 1:Ne]) * (1 / p_EE), " nS") #so far includes zero synapses, so multiply by 1/p (approx)
            println("mean excitatory synaptic weight of one assembly: ", mean(weights[1:Nmaxmembers, 1:Nmaxmembers]) * (1 / p_EE), " nS") #so far includes zero synapses, so multiply by 1/p ((approx)
            println("mean excitatory synaptic weight across assembly: ", mean(weights[1:Nmaxmembers, Nmaxmembers+1:Ne-Ne_not]) * (1 / p_EE), " nS") #so far includes zero synapses, so multiply by 1/p ((approx)
            println("mean P ->E synaptic weight: ", mean(weights[(1+Ne):(Ne+Np), 1:round(Int, (Ne - Ne_not))]) * (1 / p_EI), " nS") #multiply by 1/p
            # println("mean P ->E synaptic weight: ",mean(weights[(1+Ne):(Ne+Np),1:Ne])*(1/p_EI)," nS") #multiply by 1/p
            println("mean S ->E plastic synaptic weight: ", mean(weights[(1+Ne+Np):round(Int, (Ne + Np) + Ns * percentage_dis_inh), 1:Ne-Ne_not]) * (1 / p_EI), " nS") #multiply by 1/p (approx)
            println("mean S ->E static synaptic weight: ", mean(weights[1+round(Int, (Ne + Np) + Ns * percentage_dis_inh):Ne+Np+Ns, 1:Ne-Ne_not]) * (1 / p_EI), " nS") #multiply by 1/p (approx)

            #Note: if you see the firing rate  = 0.0Hz in the print on terminal, it's because save_partial=true and ns[] is not updated 
            #=println("mean excitatory firing rate: ",mean((ns[1:Ne] .- ns_prev[1:Ne])/((tt - tt_prev)/10000))," Hz")
            println("mean pv firing rate: ",mean((ns[Ne+1:Ne+Np] .- ns_prev[Ne+1:Ne+Np])/((tt - tt_prev)/10000))," Hz")
            println("mean sst firing rate: ",mean((ns[Ne+Np+1:Ne+Np+Ns] .- ns_prev[Ne+Np+1:Ne+Np+Ns])/((tt - tt_prev)/10000))," Hz")
            println("mean vip firing rate: ",mean((ns[Ne+Np+Ns+1:Ncells] .- ns_prev[Ne+Np+Ns+1:Ncells])/((tt - tt_prev)/10000))," Hz")
            ns_prev = copy(ns)            =#
            tt_prev = copy(tt)
        end

        ###BIOLOGICAL TIME
        t = (dt * tt)
        tprev = (dt * (tt - 1)) #time one timestep before
        ###END BIOLOGICAL TIME

        ###POISSON INPUT###
        INP = (rand(Uniform(), Ncells) .<= rx .* dt)::BitVector  #creating a train of inputs. if rx.*dt is clse to 1 (it means the stimulation frequency is high), high change that the 
        #the external input rx will create an input current INP
        ###POISSON INPUT END### =#   
        #=
                ###HOMEOSTATIC NORMALISATION### (Litwin-Kumar and Doiron 2014)  keeping the sum of each column (it is important because colomn means pre to post) constant be renormalizing (using subtraction) everything
                if mod(tt,inormalize) == 0 #excitatory synaptic normalisation
                    for cc = 1:Ne
                        sumwee = sum(weights[1:Ne,cc])
                        cond = weights[1:Ne,cc] .> 0.
                        weights[1:Ne,cc] .-= ((sumwee-sumwee0[cc])/Nee[cc]) .* cond
                        weights[findall((weights[1:Ne,cc] .< jeemin) .*cond),cc] .= jeemin
                        weights[findall((weights[1:Ne,cc] .> jeemax) .*cond),cc] .= jeemax
                    end
                end
                ###END HOMEOSTATIC NORMALISATION###
                ###START UPDATING EVERY NEURON DURING THIS TIMESTEP####
        =#
        spiked = falses(Ncells) #spike occurence
        for cc = 1:Ncells
            
            #=
            =#
            ###UPDATE Current BECAUSE OF POISSON INPUT###
            #for input spike trains

            if INP[cc]
                if cc <= Ne
                    I_ext[cc] += je0 / tauedecay
                elseif Ne < cc <= (Ne + Np)
                    I_ext[cc] += jp0 / tauedecay
                elseif (Ne + Np) < cc <= (Ne + Np + Ns)
                    I_ext[cc] += js0 / tauedecay
                else
                    I_ext[cc] += jv0 / tauedecay
                end
            end

#testing to see what happens when I increase the current at a certain point in time
            if ((tt > 5000) && (tt < 25000) && (cc > (Ne+Ns+Np)))
                I_ext[cc] += dt * 0.05 / tauedecay
            end

            ###NEURON DYNAMICS
            if t > (lastSpike[cc] + taurefrac) #only after refractory period (for neurons not in refractory period)

                ###CURRENT DYNAMICS####
                #forward euler
                Ie[cc] += -dt * Ie[cc] / tauedecay
                Ip[cc] += -dt * Ip[cc] / taupdecay
                Is[cc] += -dt * Is[cc] / tausdecay
                Iv[cc] += -dt * Iv[cc] / tauvdecay

                I_ext[cc] += -dt * I_ext[cc] / tauedecay

                ###CURRENT DYNAMICS END###

                ###LIF NEURONS###
                #forward euler
                if cc <= Ne #excitatory neuron
                    dv = -v[cc] / taue + Ie[cc] - Ip[cc] - Is[cc] + I_ext[cc]
                    v[cc] += dt * dv
                    if v[cc] > vth_e
                        spiked[cc] = true
                    end
                elseif Ne < cc <= (Ne + Np) #pv neurons
                    dv = -v[cc] / taup + Ie[cc] - Ip[cc] - Is[cc] + I_ext[cc]
                    v[cc] += dt * dv
                    if v[cc] > vth_i
                        spiked[cc] = true
                    end
                elseif (Ne + Np) < cc <= (Ne + Np + Ns)#sst neurons
                    dv = -v[cc] / taus + Ie[cc] - Iv[cc] + I_ext[cc]
                    v[cc] += dt * dv
                    if v[cc] > vth_i
                        spiked[cc] = true
                    end
                else #vip neurons
                    dv = -v[cc] / tauv + Ie[cc] - Is[cc] - Ip[cc] + I_ext[cc]
                    v[cc] += dt * dv
                    if v[cc] > vth_i
                        spiked[cc] = true
                    end
                end
                ###LIF NEURONS END###

#=
                #so if it's the right time given the frequency and it's either an E or PV, spike it
                if(should_I_spike(frequency, t) && (cc <= (Ne + Np)))

                    #if((abs(cc_prev - cc) > 10) || (cc_prev == 0))
                        #print("cc=", cc, "  t=", t, " prev_I_ext: ", I_ext[cc])
                    if(cc<=Ne)
                        #I_ext[cc] += dt * 0.05 / tauedecay
                        #spiked[cc] = true

                    else
                        #I_ext[cc] += dt * 0.05 / taupdecay
                        #spiked[cc] = true

                    end
                        #print(" post_I_ext: ", I_ext[cc], "\n")
                    #else
                    #    I_ext[cc] += dt * 5 / tauedecay
                    #end
                    cc_prev = cc

                #separate if condition to check if it's a VIP neuron since they should get diff input
                elseif(should_I_spike(frequency, t) && (cc >Ne+Np+Ns))
                    #I_ext[cc] += dt * 0.05 / tauvdecay
                    v[cc] *= 1.015
                end
=#
                
                ###UPDATE WHEN SPIKE OCCURS
                if spiked[cc] #spike occurred
                    v[cc] = vre #voltage back to reset potential
                    lastSpike[cc] = t #record last spike time
                    
                    if ns[cc] == Nspikes + 10_000 * count_add #if you finished the space, you can make up some more
                        times = hcat(times, zeros(Ncells, 10_000))
                        count_add += 1
                    end
                    if save_partial
                        if any(range_of_saving .- 10_000 .<= t .<= range_of_saving .+ 10_000) == true && ns[cc] <= Nspikes
                            ns[cc] += 1 #lists number of spikes per neuron
                            times[cc, ns[cc]] = t
                        end
                    else
                        if ns[cc] <= Nspikes #spike time are only record for ns < Nspikes
                            ns[cc] += 1 #lists number of spikes per neuron
                            times[cc, ns[cc]] = t #recording spiking times
                        end
                    end
                    
                    if cc <= Ne
                        Ie .+= (weights[cc, :]) / tauedecay
                    elseif Ne < cc <= Np+Ne
                        Ip .+= weights[cc, :] / taupdecay
                    elseif (Ne+Np) < cc <= (Np+Ne+Ns)
                        Is .+= weights[cc, :] / tausdecay
                    else
                        Iv .+= weights[cc, :] / tauvdecay
                    end

                end #end if(spiked)
            end #end if(not refractory)

                    end #end loop over cells

        if save_weights
            if save_weight_partial
                if any(t .== time_weightsave)
                    integer = round(Int, t)  #for printing reasons
                    #fid = h5open(save_weights_name * name_weight,"r+")
                    #h5write(save_weights_name * name_weight,"$integer", sparse(weights))
                    #close(fid)
                    weight_matrix_save["$integer"] = copy(sparse(weights))
                end
            else
                if (t % t_weightsave == 0) || (t == 1) #save weights
                    integer = round(Int, t)  #for printing reasons
                    #fid = h5open(save_weights_name * name_weight,"r+")
                    #h5write(save_weights_name * name_weight,"$integer", sparse(weights))
                    #close(fid)
                    weight_matrix_save["$integer"] = copy(sparse(weights))
                end
            end
        end
        if save_all
            if (save_weight_partial && (any(t .== time_weightsave))) || (t % t_weightsave == 0)
                integer = round(Int, t)  #for printing reasons
                parameters_save["x_$integer"] = copy(x)
                parameters_save["y_$integer"] = copy(y)
                parameters_save["o1_$integer"] = copy(o1)
                parameters_save["o2_$integer"] = copy(o2)
                parameters_save["r1_$integer"] = copy(r1)
                parameters_save["r2_$integer"] = copy(r2)
                parameters_save["Ie_$integer"] = copy(Ie)
                parameters_save["Ip_$integer"] = copy(Ip)
                parameters_save["Is_$integer"] = copy(Is)
                parameters_save["Iv_$integer"] = copy(Iv)
                parameters_save["I_ext_$integer"] = copy(I_ext)
                parameters_save["v_$integer"] = copy(v)
            end
        end

    end #end loop over time


    #at the end of the simulation print some useful information
    print("\r")

    println("mean excitatory synaptic weight: ", mean(weights[1:Ne, 1:Ne]), " nS") #so far includes zero synapses, so multiply by 10 (approx)
    println("mean excitatory firing rate: ", mean(ns[1:Ne] / (T / 1000)), " Hz")
    println("mean pv firing rate: ", mean(ns[(Ne+1):Ne+Np] / (T / 1000)), " Hz")
    println("mean sst firing rate: ", mean(ns[(Ne+Np+1):(Ne+Np+Ns)] / (T / 1000)), " Hz")
    println("mean vip firing rate: ", mean(ns[(Ne+Np+Ns+1):(Ncells)] / (T / 1000)), " Hz")

    ## SAVING VARIABLES
    description_save = "_both" * name * "$(zzz)" * ".jld"
    if save_weights
        JLD.save(save_weights_name * "weights" * description_save, weight_matrix_save)
    end
    if save_all
        JLD.save(save_weights_name * "parameters/" * "produced_parameters" * description_save, parameters_save)
    end

    if save_spikes
        spikes_dict = Dict()
        spikes_dict["spikes"] = sparse(times[:, 1:maximum(ns)])
        JLD.save(save_weights_name * "spikes" * description_save, spikes_dict)
        #=
        fid = h5open(save_weights_name * "spikes_$T" * name * "$(zzz)" * ".h5","w")
        h5write(save_weights_name * "spikes_$T" * name * "$(zzz)" * ".h5","spikes",times)
        close(fid)
        =#
    end
    #Input parameters
    JLD.save(save_weights_name * "parameters/" * "input_parameters" * description_save, used_param)

    #Print provided output
    println("Provided output: ")
    println(save_weights_name * "weights" * description_save)
    println(save_weights_name * "spikes" * description_save)
    if save_all
        println(save_weights_name * "produced_parameters" * description_save)
    end
    #write(save_weights_name * "spike_ateachtime_$T", spike_matrix)   #saving bitmatrix

    elapsed_time = time() -  time1
    print("Elapsed time value: ", elapsed_time)

end



###Parameters to vary###
function parameters()

    percentage_dis_inh = 0.8 #meaning 90% DisNet, 10% Inhnet

    ### Plasticity mechanisms ###
    save_partial = false #used to only partially save the spike events for RAM reasons

    PV_symmetric = true
    SST_hebb = true #always true in this case

    name = "_sst_hebb_"#2' #hebbian pairwise STDP (asymmetric hebbian)

    stim_time = 1000
    gap_time = 1000

    N = 5000 #Number of neurons
    n_E = 0.8 #Proportion of excitatory neurons
    n_p = 0.1 #Proportion of PV neurons
    n_s = 0.05 #Proportion of SST neurons
    n_v = 0.05 #Proportion of VIP neurons

    Ne = Int(N * n_E)
    Np = Int(N * n_p)
    Ns = Int(N * n_s)
    Nv = Int(N * n_v)

    #isn't Ncells just equal to N always? - M
    Ncells = Ne + Np + Ns + Nv

    #Connection probabilities
    p_EE = 0.1 #E to E
    p_EI = 0.1 # I to E
    p_IE = 0.1 #E to I
    p_II = 0.1 #I to I

    #Weights for initialisation
    jee0 = 0.35 #initial E to E in mV
    jpe, jse, jve = 4 * jee0, 0.2 * jee0, 0.2 * jee0 #weights E to I
    jep0, jpp, jvp = 12 * jee0, 9 * jee0, 0.2 * jee0# Weights P to else in mV 
    jps, jvs, jes0 = 7 * jee0, 9 * jee0, 6 * jee0 #Weights S to else in mV
    #jps, jvs, jes0=4*jee0, 6*jee0, 4*jee0 #Weights S to else in mV
    jsv = 3 * jee0 #Weights V to S in mV

    ###Amplitude currents for background spike train ###
    j0 = 0.35# input current in mV
    je0, jp0, js0, jv0 = 1 * j0, 0.7 * j0, 0.25 * j0, 0.27 * j0  #background input to each neuron population 

    r_ext = 5.0 #kHz, external input rate for Poisson process
    stim_rate = 5.0 #additional input during training in kHz

    #Clustering
    Nmaxmembers = 200 #number of neurons in an assembly
    n_bg = 0.25#0.1 #fraction of exc neurons not in an assembly
    Ne_not = Int(Ne * n_bg)
    Npop = Int(N * n_E * (1 - n_bg) ÷ Nmaxmembers) #number of assemblies

    range_plast_sst = collect(1+Ne+Np:round(Int, (Ne + Np) + Ns * percentage_dis_inh)) #for SST

    ### Simulation details ###
    T = 10_000#ms simulation time

    N_trials = 18# repetition of stimuli for assemblies formation

    if save_partial
        Nspikes = round(Int, (T / 4) / 5) # use when you want to save memory
    else
        Nspikes = round(Int, T / 4) #maximum number of spikes to record per neuron    
    end

    ###membrane dynamics###
    taue, taup, taus, tauv = 20, 20, 20, 20 #  time constants ms 

    vth_e = 10.0 #spike voltage threshold mV
    vth_i = 10.0 #mV
    vre = 0.0 #reset potential mV

    taurefrac = 1. #absolute refractory period in ms

    ###connectivity###  (Jiang 2015 and Pfeffer 2013)
    #Ncells = Ne+Np+Ns+Nv

    tauedecay = 3.
    taupdecay = 4.
    tausdecay = 5.
    tauvdecay = 5. #in ms

    jeemin = jee0 * 0.1 #0.1 #min E to E mV
    jeemax = jee0 * 20

    jepmin = 1 * jep0# min P to E mV
    jepmax = 20 * jep0 #max P to E mV (this seems high)

    jesmin = 1 * jes0 #min S to E mV
    jesmax = 20 * jes0 #max S to E mV


    ###triplet stdp###  (everything from Pfister and Gerstner 2006) === PLASTICITY RULE!
    #Parameters with original Pfister Gerstner values
    tau_plus = 16.8 #detector for pairwise presynaptic events, time constant in ms
    tau_minus = 33.7 #detector for pairwise postsynaptic events, time constant in ms
    tau_x = 101.0 #detector for triplet presynaptic events, time constant in ms
    tau_y = 125.0 #detector for triplet postsynaptic events, time constant in ms

    A2_plus = 5e-10 #pairwise potentiation term, in mV
    A2_minus = 7.0e-3 #pairwise depression term, in mV
    A3_plus = 6.2e-3 #triplet potentiation term, in mV
    A3_minus = 2.3e-4 #triplet depression term, in mV

    ###inhibitory stdp###
    tau_ipv = 20 #time constant of pre- and postsynaptic detector, in ms
    tau_isst = 30 #time constant of pre- and postsynaptic detector, in ms

    eta_pv = 0.1# homeostatic iSTDP learning rate in mV
    a_pre = 0.81# homeostatic asymmetric-STDP learning rate in mV
    a_post = 0.9# homeostatic asymmetric-STDP learning rate in mV
    r0_pv = 0.003# E target rate (kHz)

    ###simulation details###
    dt = 0.1 #integration timestep in ms
    dtnormalize = 20 #200 #homeostatic normalisation, every dtnormalize ms

    train_start = 10_000 #ms start of training
    t_weightsave = 10_000 #save weights every t_weightsave ms
    save_weights = true #want to save weights
    save_spikes = true #want to save spikes
    save_all = true
    save_weight_partial = false
    time_weightsave = [1, 10_000, 100_000, 250_000, 500_000, 700_000, 800_000, 900_000, T]


    #save_weights_name="/home/ge74coy/mnt/naspersonal/training/"
    save_weights_name = "C:/Users/mijan/Desktop/Uni PDFs/TUM/simulation_saves/"
    
    # all values found at https://www.pnas.org/doi/10.1073/pnas.2311040121#supplementary-materials
    #Constants for various networks; units in ms and a.u.

    all_param_string = "T, N_trials, stim_rate, train_start, t_weightsave, time_weightsave, 
    save_weights, save_spikes, save_weights_name, save_partial, save_all, save_weight_partial,
    je0, jp0, js0, jv0, r_ext, SST_hebb, PV_symmetric,
    Ne, Np, Ns, Nv, Ncells, Nspikes, Npop, Nmaxmembers, Ne_not,
    jee0,jpe,jse,jep0,jpp, jvp,jes0,jps,jsv,jvs,jve,percentage_dis_inh,
    p_EE,p_EI,p_IE,p_II,name, stim_time , gap_time, range_plast_sst,
    taue,taup,taus,tauv, vth_e, vth_i, vre, taurefrac, tauedecay, taupdecay, tausdecay, tauvdecay,
    jeemin, jeemax, jepmin, jepmax, jesmin, jesmax,
    tau_ipv,tau_isst, tau_plus, tau_minus, tau_x, tau_y, A2_minus, A2_plus, A3_minus, A3_plus, 
    eta_pv, r0_pv, a_pre, a_post, dt, dtnormalize"
    all_param_tuple = (T, N_trials, stim_rate, train_start, t_weightsave, time_weightsave,
        save_weights, save_spikes, save_weights_name, save_partial, save_all, save_weight_partial,
        je0, jp0, js0, jv0, r_ext, SST_hebb, PV_symmetric,
        Ne, Np, Ns, Nv, Ncells, Nspikes, Npop, Nmaxmembers, Ne_not,
        jee0, jpe, jse, jep0, jpp, jvp, jes0, jps, jsv, jvs, jve, percentage_dis_inh,
        p_EE, p_EI, p_IE, p_II, name, stim_time, gap_time, range_plast_sst,
        taue, taup, taus, tauv, vth_e, vth_i, vre, taurefrac, tauedecay, taupdecay, tausdecay, tauvdecay,
        jeemin, jeemax, jepmin, jepmax, jesmin, jesmax,
        tau_ipv, tau_isst, tau_plus, tau_minus, tau_x, tau_y, A2_minus, A2_plus, A3_minus, A3_plus,
        eta_pv, r0_pv, a_pre, a_post, dt, dtnormalize)

    used_param = save_param(all_param_string, all_param_tuple)

    return T, N_trials, stim_rate, train_start, t_weightsave, time_weightsave,
    save_weights, save_spikes, save_weights_name, save_partial, save_all, save_weight_partial,
    je0, jp0, js0, jv0, r_ext, SST_hebb, PV_symmetric,
    Ne, Np, Ns, Nv, Ncells, Nspikes, Npop, Nmaxmembers, Ne_not,
    jee0, jpe, jse, jep0, jpp, jvp, jes0, jps, jsv, jvs, jve, percentage_dis_inh,
    p_EE, p_EI, p_IE, p_II, name, stim_time, gap_time, range_plast_sst,
    taue, taup, taus, tauv, vth_e, vth_i, vre, taurefrac, tauedecay, taupdecay, tausdecay, tauvdecay,
    jeemin, jeemax, jepmin, jepmax, jesmin, jesmax,
    tau_ipv, tau_isst, tau_plus, tau_minus, tau_x, tau_y, A2_minus, A2_plus, A3_minus, A3_plus,
    eta_pv, r0_pv, a_pre, a_post, dt, dtnormalize, used_param
end

sim(1)
