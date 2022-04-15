--current_dir = string.gsub(debug.getinfo(1).source, "^@(.+/)[^/]+$", "%1")
--local consts = dofile(current_dir .. "consts.lua")

ikEnv = 1
res_time = 15000
joint_test = 3
tf = 2000
UR5_base = '.'
UR5_tip = './UR5_tip'
UR5_target = './UR5_target'

function dot_product(q1, q2)
    local num = q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3] + q1[4]*q2[4]
    local den1 = math.sqrt(q1[1]^2 + q1[2]^2 + q1[3]^2 + q1[4]^2)
    local den2 = math.sqrt(q1[1]^2 + q1[2]^2 + q1[3]^2 + q1[4]^2)
    local ret = num/(den1*den2)
    
    if ret > 1 or ret < -1 then
        ret = 0
    else
        ret = math.acos(ret)
    end
    
    return ret
end

function subSetArray(original, beginning, ending)
    local rc = {}
    for i=beginning, ending, 1 do
        table.insert(rc, original[i])
    end
    return rc
end

function print_array(pr_arr)
    for i=1, #pr_arr, 1 do
        sim.addLog(sim.verbosity_scriptinfos, pr_arr[i])
    end
    sim.addLog(sim.verbosity_scriptinfos, "+-+-+-+-+-+-")
end

function IK_init()
    simBase=sim.getObject(UR5_base)
    simTip=sim.getObject(UR5_tip)
    simTarget=sim.getObject(UR5_target)
    -- create an IK environment:
    ikEnv=simIK.createEnvironment()
    -- create an IK group: 
    ikGroup_undamped=simIK.createIkGroup(ikEnv)
    -- set its resolution method to undamped: 
    simIK.setIkGroupCalculation(ikEnv,ikGroup_undamped,simIK.method_pseudo_inverse,0,6)
    -- create an IK element based on the scene content: 
    simIK.addIkElementFromScene(ikEnv,ikGroup_undamped,simBase,simTip,simTarget,simIK.constraint_pose)
    -- create another IK group: 
    ikGroup_damped=simIK.createIkGroup(ikEnv)
    -- set its resolution method to damped: 
    simIK.setIkGroupCalculation(ikEnv,ikGroup_damped,simIK.method_damped_least_squares,1,99)
    -- create an IK element based on the scene content: 
    simIK.addIkElementFromScene(ikEnv,ikGroup_damped,simBase,simTip,simTarget,simIK.constraint_pose) 
    ik_activated = true
end

function setJointPositions(positions)
    for i=1,#jointHandles,1 do
        sim.setJointTargetPosition(jointHandles[i], positions[i])
    end
end

function getJointPositions()
    for i=1,#jointHandles,1 do
        current_joints[i] = sim.getJointPosition(jointHandles[i])
    end
end

function getTorques()
    for i=1,#jointHandles,1 do
        current_torque[i] = sim.getJointForce(jointHandles[i])
    end
end

function store_initial_information()
    getJointPositions()
    stored_joints = current_joints
    print_array(stored_joints)
    stored_sphere = sim.getObjectPose(targetSphere, -1)
    sphere_position = stored_sphere
    
    posT = subSetArray(stored_sphere, 1, 3)
    oriT = subSetArray(stored_sphere, 4, 7)
end

function quaternion_interpolation(qt, gm, t_i)
    local qr = {}
    local ti = t_i/tf
    
    for i=1, 4, 1 do
        qr[i] = math.sin((1-ti)*gm)/math.sin(gm)*qt[i] + math.sin(ti*gm)/math.sin(gm)*qt[i]
    end
    
    return qr
end

function linear_interpolation(ti, target, orig)
    local st = ti/tf
    local xt = {}
    
    for i=1, #target, 1 do
        xt[i] = orig[i] + (target[i]-orig[i])*st
    end
        
    return xt
end

function restarting_ball()
    local currentPose = sim.getObjectPose(targetSphere, -1)
    
    linearOrien = false
    
    position0 = subSetArray(currentPose, 1, 3)
    ori0 = subSetArray(currentPose, 4, 7)
    
    gamma = dot_product(ori0, oriT)
    
    if (gamma < 0) then
        local auxori = {}
        for i=1, 4, 1 do
            auxori[i] = ori0[i] * -1
        end
        gamma = dot_product(auxori, oriT)
    end
    
    if gamma == 0 or not gamma then
        linearOrien = true
        sim.addLog(sim.verbosity_scriptinfos, "linear")
    end
     sim.addLog(sim.verbosity_scriptinfos, gamma)
    time_inter_in = sim.getSystemTimeInMs(-1)
    restarting = true
end

function compute_velocity()
    vel_time = actual_time - previous_time
    previous_time = actual_time
    
    if (not(1000 == previous_joints[1])) then
        for i=1, #current_joints, 1 do
            currentVel[i] = (current_joints[i] - previous_joints[i])/vel_time
        end
    end
    previous_joints = current_joints
end

function changeAngle(posi, aVeloc)
    --toQuaternion(angles[3], angles[2], angles[1])
    posi[4] = posi[4] + (aVeloc[1]*posi[7]+aVeloc[3]*posi[5]-aVeloc[2]*posi[6])/2
    posi[5] = posi[5] + (aVeloc[2]*posi[7]-aVeloc[3]*posi[4]+aVeloc[1]*posi[6])/2
    posi[6] = posi[6] + (aVeloc[3]*posi[7]+aVeloc[2]*posi[4]-aVeloc[1]*posi[5])/2
    posi[7] = posi[7] + (-aVeloc[1]*posi[4]-aVeloc[2]*posi[5]-aVeloc[3]*posi[6])/2
    
    return posi
end
function restart_simulation(data)
    if data.data then
        restarting_ball()
    end
end

function velocitySphere_cb(velocity)
    if (actual_time > max_delay) and not restarting then
            
        local pos = sim.getObjectPose(targetSphere, -1)
        sphere_position = pos
        local angularVelocity = subSetArray(velocity.data, 4, 6)

        for i=1, 3, 1 do
            pos[i] = pos[i] + (maxVel*velocity.data[i])/maxRec
        end
        
        for i = 1, 3, 1 do
            angularVelocity[i] =(maxAngularVel*angularVelocity[i])/maxRec
        end
        
        pos = changeAngle(pos, angularVelocity)

        if (-1 == sim.setObjectPose(targetSphere, -1, pos)) then
            sim.addLog(sim.verbosity_scriptinfos, "Error")
        end
                    
    end
end

function add_time_iter(tim, iter, orig_arr)
    local rc = {}
    rc[1] = iter
    rc[2] = tim
    
    for i=1, #orig_arr, 1 do
        rc[#rc+1] = orig_arr[i]
    end
    
    return rc
end

function sysCall_init()
    name_UR_joints = 'UR5_joint'
    name_UR5_sphere_aux = "UR5_manipSphere"
    move_topic = '/move_command'
    simulation_reset_topic = '/ur5_simulation/reset'
    simulation_position_topic = '/ur5_simulation/jointConfig'
    simulation_tool_position_topic = '/ur5_simulation/ToolPosition'    
    simulation_torque_topic = '/ur5_simulation/torques'
    simulation_speed_topic = '/ur5_simulation/speeds'
    ik_activated = false
    linearOrien = false
    restarting = false
    rep = 0
    sphere_position = {}
    
    delay_activate_IK = 300
    
    max_delay = 500
    
    delay= 0
    
    maxVel = 0.02
    maxAngularVel = 0.03
    maxRec = 1024

    targetSphere = sim.getObjectHandle(name_UR5_sphere_aux)
    
    time_r = sim.getSystemTimeInMs(-1)
    previous_time = 0

    currentVel={0,0,0,0,0,0}
    current_joints={0,0,0,0,0,0}
    current_torque={0,0,0,0,0,0}
    previous_joints={1000,1000,1000,1000,1000,1000}
    currentPositions={0,0,0,0,0,0}
    jointHandles={}

    for i=1,6,1 do
        jointHandles[i]=sim.getObjectHandle(name_UR_joints .. i)
    end
    mas_in, off, mul = sim.getJointDependency(jointHandles[joint_test])
    
    store_initial_information()
    IK_init()
    
    if simROS then
        sim.addLog(sim.verbosity_scriptinfos, "Ros interface was found.")
        
        move_sub = simROS.subscribe(move_topic, 'std_msgs/Float32MultiArray','velocitySphere_cb')
        sim_pub_reset = simROS.subscribe(simulation_reset_topic, 'std_msgs/Bool','restart_simulation')
        sim_pub_position = simROS.advertise(simulation_position_topic, 'std_msgs/Float32MultiArray')
        sim_pub_torque = simROS.advertise(simulation_torque_topic, 'std_msgs/Float32MultiArray')
        sim_pub_speed = simROS.advertise(simulation_speed_topic, 'std_msgs/Float32MultiArray')
        sim_pub_tool_position = simROS.advertise(simulation_tool_position_topic, 'std_msgs/Float32MultiArray')
    else
        sim.displayDialog('Error','The RosInterface was not found.',sim.dlgstyle_ok,false,nil,{0.8,0,0,0,0,0},{0.5,0,0,1,1,1})
    end
end

function sysCall_actuation()

    if (restarting) then
        time_inter = sim.getSystemTimeInMs(time_inter_in)
        pose = {}
        orient = {}
        
        if (tf > time_inter) then
            pos_i = linear_interpolation(time_inter, posT, position0)
            
            if linearOrien then
                orient = linear_interpolation(time_inter, oriT, ori0)
            else
                orient = quaternion_interpolation(oriT, gamma, time_inter)
            end
            
            pose = pos_i
            for i=1, #orient, 1 do
                pose[#pose+1]=orient[i]
            end
            if (-1 == sim.setObjectPose(targetSphere, -1, pose)) then
                sim.addLog(sim.verbosity_scriptinfos, "Error")
            end
        else
            
            time_r = sim.getSystemTimeInMs(-1)
            previous_time = time_r
            max_delay = 400
            restarting = false
            rep = rep + 1
        end
    end
    
    if (ik_activated) then
        -- try to solve with the undamped method:
        if simIK.applyIkEnvironmentToScene(ikEnv,ikGroup_undamped,true)==simIK.result_fail then 
            -- the position/orientation could not be reached.
            -- try to solve with the damped method:
            simIK.applyIkEnvironmentToScene(ikEnv,ikGroup_damped)
            -- We display a IK failure report message:
            sim.addLog(sim.verbosity_scriptwarnings,"IK solver failed.") 
        end
    end
    
end

function sysCall_sensing()
    actual_time = sim.getSystemTimeInMs(time_r)
    if not restarting then
        if (actual_time > max_delay) then
            getJointPositions()
            getTorques()
            compute_velocity()
            
            aux_pos = add_time_iter(actual_time, rep, current_joints)
            aux_torque = add_time_iter(actual_time, rep, current_torque)
            aux_vel = add_time_iter(actual_time, rep, currentVel)
            aux_sphere_pos = add_time_iter(actual_time, rep, sphere_position)
            
            if simROS then
                simROS.publish(sim_pub_position,{data=aux_pos})
                simROS.publish(sim_pub_torque,{data=aux_torque})
                simROS.publish(sim_pub_speed,{data=aux_vel})
                simROS.publish(sim_pub_tool_position,{data=aux_sphere_pos})
            end
                    
        end
        
        if ((actual_time > delay_activate_IK) and not(ik_activated)) then
            IK_init()
        end
        
        
    end
    sim.addLog(sim.verbosity_scriptinfos, actual_time)
    sim.addLog(sim.verbosity_scriptinfos, "-------")
end

function sysCall_cleanup()
    if simROS then
        simROS.shutdownSubscriber(move_sub)
        simROS.shutdownSubscriber(sim_pub_reset)
        simROS.shutdownPublisher(sim_pub_position)
        simROS.shutdownPublisher(sim_pub_torque)
        simROS.shutdownPublisher(sim_pub_speed)
        simROS.shutdownPublisher(sim_pub_tool_position)
    end
    -- erase the IK environment: 
    simIK.eraseEnvironment(ikEnv) 
end

-- See the user manual or the available code snippets for additional callback functions and details

