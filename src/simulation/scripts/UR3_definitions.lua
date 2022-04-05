--current_dir = string.gsub(debug.getinfo(1).source, "^@(.+/)[^/]+$", "%1")
--local consts = dofile(current_dir .. "consts.lua")


function move_JointPosition(positions)
    for i=1,number_joints,1 do
         sim.setJointTargetPosition(jointHandles[i], positions[i])
    end
end

-- Callback function for moving the joints
function setMotorPositions_cb(msg)
    data = msg.data

    move_JointPosition(data)
end

function getJointPositions()
    for i=1,number_joints,1 do
        currentPositions[i] = simGetJointPosition(jointHandles[i])
    end
end

function getTorques()
    for i=1,number_joints,1 do
        currentTorque[i] = simGetJointForce(jointHandles[i])
    end
end

function getVelocities()
    for i=1,number_joints,1 do
        res,currentVel[i] = simGetObjectFloatParameter(jointHandles[i],2012)
    end
end

--[[
Initialization: Called once at the start of a simulation
--]]
if (sim_call_type==sim.childscriptcall_initialization) then

    name_UR_joints = 'UR3_joint'
    number_joints = 6

    max_velocity = 180
    max_acceleration = 40
    max_jerk = 80

    jointHandles={-1,-1,-1,-1,-1,-1}

    for i=1,number_joints,1 do
        jointHandles[i]=sim.getObjectHandle(name_UR_joints .. i)
    end

    vel=max_velocity
    accel=max_acceleration
    jerk=max_jerk
    currentVel={0,0,0,0,0,0,0}
    currentTorque={0,0,0,0,0,0,0}
    currentAccel={0,0,0,0,0,0,0}
    currentPositions={0,0,0,0,0,0,0}

    --maxVel={vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180,vel*math.pi/180}
    --maxAccel={accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180,accel*math.pi/180}
    --maxJerk={jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180,jerk*math.pi/180}
    
    for i=1,number_joints,1 do
        sim.setObjectFloatParameter(jointHandles[i], sim_jointfloatparam_upper_limit, vel*math.pi/180)
    end

    -- Check if the required ROS plugin is loaded
    
    if simROS then
        sim.addLog(sim.verbosity_scriptinfos, "Ros interface was found.")
        MotorSub = simROS.subscribe('/ur3/move_command', 'std_msgs/Float32MultiArray','setMotorPositions_cb')
        simROS.subscriberTreatUInt8arrayAsString(MotorSub)

        jointPositionsPub=simROS.advertise('/ur3/joint_positions','std_msgs/Float32MultiArray')
        jointTorquesPub=simROS.advertise('/ur3/joint_torques','std_msgs/Float32MultiArray')
        jointVelocitiesPub=simROS.advertise('/ur3/joint_velocities','std_msgs/Float32MultiArray')
    else
        sim.displayDialog('Error','The RosInterface was not found.',sim.dlgstyle_ok,false,nil,{0.8,0,0,0,0,0},{0.5,0,0,1,1,1})
    end
end

if (sim_call_type==sim.childscriptcall_sensing) then
    getJointPositions()
    getTorques()
    getVelocities()

    --print("Current velocity: " .. currentVel[1], currentVel[2], currentVel[3], currentVel[4], currentVel[5], currentVel[6])
    --print("currentTorque: " .. currentTorque[1], currentTorque[2], currentTorque[3], currentTorque[4], currentTorque[5], currentTorque[6])

    if simROS then
        simROS.publish(jointPositionsPub,{data=currentPositions})
        simROS.publish(jointVelocitiesPub,{data=currentVel})
        simROS.publish(jointTorquesPub,{data=currentTorque})
    end
    
end

if (sim_call_type==sim.childscriptcall_cleanup) then
    

    -- Terminate remaining local notes
    if simROS then
        simROS.shutdownSubscriber(MotorSub)
        simROS.shutdownPublisher(jointTorquesPub)
        simROS.shutdownPublisher(jointVelocitiesPub)
        simROS.shutdownPublisher(jointPositionsPub)
    end
end
