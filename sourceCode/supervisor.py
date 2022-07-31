"""
Client of V-REP simulation server

Before running the script, the simulation must be running on VREP (the robot script initializes the server).
Alternatively, initialize the server with the LUA command simExtRemoteApiStart(portNumber) and remove server
initialization from robot script.

This scrip implements a bug "go to destination" algorithm. The robot is equiped with a sonar ring, a GPS
and a compass and knows the position of a flag that must be reached. The robots tries to move towards
the flag, however, whenever a obstacle is encountered, it switches to a wall following behavior until
it has a clear line of sight (considering the reach of its sensors) to the flag.

For some scenarios, a floating (floating so as to not be detected by the sonars) waypoint was manually
defined in order to avoid some local minima "rooms". Whenever the waypoint is in between the robot and the
flag, the waypoint is the objective, else the flag is the objective.
"""

import numpy as np
import sys

try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

#------------------------------ GLOBAL VARIABLES ------------------------------#

v0 = 2.0         # default speed
turnSpeed = 1.0  # default turning speeds
#------------------------------------------------------------------------------#


#------------------------------------ UTILS -----------------------------------#

def getHandle( path ):
    """
    Get object handle from path.
    Arguments:
    - path: object path
    Return:
    - handle 
    """
    error, handle = sim.simxGetObjectHandle( clientID, path, sim.simx_opmode_oneshot_wait )
    if error != 0:
        print ( '{} handle not found!'.format(path) )
    else:
        print ( 'Connected to {}!'.format(path) )
    return handle

def readPositionSignal( reference, verbose=False ):
    """
    Read x, y and z coordinates from appropriate signals. Assumes the signals have been appropriately
    defined as "<coordinate><Reference>Pos"
    Arguments:
    - reference: whose coordinates to read
    - verbose: whether to print coordinates
    Return:
    - pos: vector containing x, y and z coordinates
    """
    pos = [0,0,0]
    error = [0,0,0]
    error[0], pos[0] = sim.simxGetFloatSignal(clientID, "x"+reference+"Pos", sim.simx_opmode_buffer)
    error[1], pos[1] = sim.simxGetFloatSignal(clientID, "y"+reference+"Pos", sim.simx_opmode_buffer)
    error[2], pos[2] = sim.simxGetFloatSignal(clientID, "z"+reference+"Pos", sim.simx_opmode_buffer)

    if verbose:
        if sum(error) != 0:
            print( "Failed to get {} position! Error codes: {}".format(reference, error) )
        else:
            print( "{} GPS: x={:.4f} y={:.4f} z={:.4f}".format( reference, *pos ) )

    error = sum(error)
    return error, pos

def getPioneerPose():
    """
    Computes the robot pose.
    Return:
    - error: whether there was an error reading the signals
    - pose: dictionary with keys: 
                "Position": robot position
                "Orientation": robot orientation ([0, 2pi])
    """
    errorPioneerPos, pioneerPos = readPositionSignal("Pioneer")
    errorPioneerOrient, pioneerOrient = sim.simxGetFloatSignal(clientID, "compassAngle", sim.simx_opmode_buffer)

    pose = {"Position": [0.0, 0.0, 0.0],
            "Orientation": 0.0}
    error = errorPioneerPos+errorPioneerOrient

    if not error:   
        pose["Position"] = pioneerPos
        pose["Orientation"] = pioneerOrient if pioneerOrient>=0 else 2*np.pi+pioneerOrient # [0, 2pi]

    return error, pose

def positionRelativeToPoint( reference, pointPos, errorPointPos, pose, verbose=False ):
    """
    Computes the robot pose with respect to a given reference.
    Arguments:
    - reference: point name
    - pointPos: point with respect to which the pose must be computed ( [x, y] or [x, y, z] )
    - errorPointPos: whether the position given is corrupted or not
    - pose: dictionary with keys:
                "Position": robot position
                "Orientation": robot orientation
    Return:
    - error: whether there was an error reading the signals
    - pose: dictionary with additional keys: 
                "DistanceTo<reference>": distance from robot to point
                "AngleTo<reference>": angle between the line through robot and point and the scene's x axis ([0, 2pi])
                "RotationTo<reference>": angle between robots orientation and the line through robot and point ([-pi, pi])
                "<reference>Position": point position
    """

    pose.update( {"DistanceTo"+reference:   0.0,
                    "AngleTo"+reference:    0.0,
                    "RotationTo"+reference: 0.0,
                    reference+"Position":   0.0} )

    if not errorPointPos:

        pose[reference+"Position"] = pointPos

        # compute distance to point
        pose["DistanceTo"+reference] = np.linalg.norm( np.array( pose["Position"][0:2] ) - np.array( pointPos[0:2] ) )

        # compute angle to point
        pointTX = pointPos[0] - pose["Position"][0]
        pointTY = pointPos[1] - pose["Position"][1]

        pose["AngleTo"+reference] = np.arctan2( pointTY, pointTX )
        pose["AngleTo"+reference] = pose["AngleTo"+reference] if pose["AngleTo"+reference]>=0 else 2*np.pi+pose["AngleTo"+reference] # [0, 2pi]

        # compute angular difference between orientation and relative position
        pose["RotationTo"+reference] = pose["AngleTo"+reference] - pose["Orientation"]

        if pose["RotationTo"+reference] > np.pi:
            pose["RotationTo"+reference] -= 2.0*np.pi
        elif pose["RotationTo"+reference] < -np.pi:
            pose["RotationTo"+reference] += 2.0*np.pi

        if verbose:
            print( "Orientation: {:.2f} -- Angle to {}: {:.2f} -- Orientation to {}: {:.2f} -- Distance to {}: {:.2f}".format( pose["Orientation"], reference, pose["AngleTo"+reference], reference, pose["RotationTo"+reference], reference, pose["DistanceTo"+reference] ) )
    
    return pose

def moveTowardsFlag( point, path, verbose=False ):
    """
    Computes motor speeds needed to orient and move robot towards flag.
    Parameters:
    - point: current point on path
    - path: path of points
    Return:
    - vLeft: left motor speed
    - vRight: right motor speed
    - reachedFlag: whether the destination has been reached or not
    - pose: robot pose
    - reference: which objective the robot is pursuing
    """

    print("Waypoint #{}: {}".format(point, path[point]))

    # distance within which it is considered the robot has reached the desired point
    toleranceFlag = 1.0    
    tolerancePoint = 5.0

    _, pose = getPioneerPose()
    errorFlagPos, flagPos = readPositionSignal( "Flag" )
    pose = positionRelativeToPoint( "Flag", flagPos, errorFlagPos, pose, verbose )
    pose = positionRelativeToPoint( "Point", path[point], False, pose, verbose )

    vLeft = 0
    vRight = 0
    reachedFlag = False

    if( pose["DistanceToFlag"] <= toleranceFlag ):
        if verbose:
            print("[SUCCESS] Reached destination!")
        reachedFlag = True  
    else:     
        if( pose["DistanceToPoint"] <= tolerancePoint ):
            if point < len(path)-1:
                point += 1

        vRight = v0*pose["RotationToPoint"]/np.pi
        vLeft = -vRight

        vLeft += v0
        vRight += v0
        

    return vLeft, vRight, reachedFlag, pose, point

def turnRightInPlace( verbose=False ):
    vLeft = turnSpeed
    vRight = -turnSpeed
    if verbose:
        print(" -- Turn right in place") 
    return vLeft, vRight

def turnLeftInPlace( verbose=False ):
    vLeft = -turnSpeed
    vRight = turnSpeed
    if verbose:
        print(" -- Turn left in place") 
    return vLeft, vRight

def driveForward( verbose=False ):
    vLeft = vRight = v0
    if verbose:
        print(" -- Drive forward") 
    return vLeft, vRight

def getSonarReadings( verbose=False ):
    """
    Get proximity sensor readings.
    """

    # Pioneer sonar distribution:
    #     2  3  4  5
    #  1 /          \ 6
    #  0 |          | 7
    #    |          |
    # 15 |          | 8
    # 14 \          / 9
    #     13 12 11 10

    dist = []

    noDetectionDist=0.5
    maxDetectionDist=0.2
    detect=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]

    for i in range(16):
        error, state, coord, _, _ = sim.simxReadProximitySensor(clientID, sensorHandle[i],sim.simx_opmode_buffer)

        if error or state == 0:
            dist += [ np.inf ]
        else:
            dist += [ np.linalg.norm( coord[0:-1] ) ]   # compute 2D distance to obstacle

        if verbose:
            print("{}({:.2f})".format( i, dist[-1] ), end=" ")

        if error == 0:
            d = coord[2]
            if state > 0 and d < noDetectionDist:
                if d < maxDetectionDist:
                    d = maxDetectionDist

                detect[i] = 1-((d-maxDetectionDist) / (noDetectionDist-maxDetectionDist))
            else:
                detect[i] = 0
        else:
            detect[i] = 0

    if verbose:
        print( end="\n" )
    
    return dist, detect

def braitenberg( vLeft, vRight, pose, reference, verbose=False ):
    """
    Braitenberg obstacle avoidance behavior.
    Arguments:
    - vLeft: left motor speed computed by go-to-destination module
    - vRight: right motor speed computed by go-to-destination module
    - pose: dictionary computed by positionRelativeToFlag()
    - reference: which objective the robot is pursuing
    Return:
    - vLeft: left motor speed
    - vRight: right motor
    """

    uncertainty = 0.1
    conversionFactor = 10.0

    braitenbergL=[-0.2,-0.4,-0.6,-0.8,-1,-1.2,-1.4,-1.6, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    braitenbergR=[-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    dist, detect = getSonarReadings( verbose )

    # frontObstacle = ( (dist[3] != np.inf) or (dist[4] != np.inf) )
    frontObstacle = True 

    if frontObstacle:
        # if conversionFactor*np.min( [dist[3], dist[4]] ) < pose["DistanceToFlag"] - uncertainty:
        if conversionFactor*np.min( dist ) < pose["DistanceToFlag"] - uncertainty:

            vLeft = v0
            vRight = v0

            for i in range(16):
                vLeft  = vLeft  + braitenbergL[i] * detect[i]
                vRight = vRight + braitenbergR[i] * detect[i]

        elif verbose:
            print( " -- Go to destination (flag closer then obstacle)" )
    
    return vLeft, vRight



def followWall( vLeft, vRight, pose, reference, verbose=False ):
    """
    Wall-folowing behavior. Superseeds go-to-destination bahevior whenever there is a obstacle on line-of-sight.
    Arguments:
    - vLeft: left motor speed computed by go-to-destination module
    - vRight: right motor speed computed by go-to-destination module
    - pose: dictionary computed by positionRelativeToFlag()
    - reference: which objective the robot is pursuing
    Return:
    - vLeft: left motor speed
    - vRight: right motor
    """

    uncertainty = 0.1
    conversionFactor = 10.0

    # Pioneer sonar distribution:
    #     2  3  4  5
    #  1 /          \ 6
    #  0 |          | 7
    #    |          |
    # 15 |          | 8
    # 14 \          / 9
    #     13 12 11 10
    
    dist, _ = getSonarReadings( verbose )

    leftWall = ( dist[0] != np.inf )
    rightWall = ( dist[7] != np.inf )
    frontWall = ( (dist[3] != np.inf) or (dist[4] != np.inf) )

    if verbose:
        print( "L: {} -- F: {} -- R: {}".format(leftWall, frontWall, rightWall), end="" )

    if frontWall:
        if conversionFactor*np.min( [dist[3], dist[4]] ) < pose["DistanceToFlag"] - uncertainty:
            if dist[2] < dist[5]:
                vLeft, vRight = turnRightInPlace( verbose )
            elif dist[5] < dist[2]:
                vLeft, vRight = turnLeftInPlace( verbose )
            elif pose["RotationTo"+reference] > 0:    # reference to the left
                vLeft, vRight = turnLeftInPlace( verbose )
            else:                                     # reference to the right
                vLeft, vRight = turnRightInPlace( verbose )
        elif verbose:
            print( " -- Go to destination (flag closer then obstacle)" )
    elif leftWall:
        if conversionFactor*dist[0] < pose["DistanceToFlag"] - uncertainty:
            if pose["RotationTo"+reference] > 0:      # reference to the left
                vLeft, vRight = driveForward( verbose )
            elif verbose:
                print( " -- Go to destination (no obstacles on line of sight)")       
        elif verbose:
            print( " -- Go to destination (flag closer then obstacle)" )     
    elif rightWall:
        if conversionFactor*dist[7] < pose["DistanceToFlag"] - uncertainty:
            if pose["RotationTo"+reference] < 0:      # reference to the right
                vLeft, vRight = driveForward( verbose )
            elif verbose:
                print( " -- Go to destination (no obstacles on line of sight)")  
        elif verbose:
            print( " -- Go to destination (flag closer then obstacle)" )
    else: # no obstacle: go-to-destination
        if verbose:
            print( " -- Go to destination (no obstacles)")

    return vLeft, vRight

def updateSpeed( vLeft, vRight, verbose=False ):
    """
    Updated robot speed.
    Arguments:
    - vLeft: left motor speed
    - vRIght: right motor speed
    """
    if verbose:
        print("vl: {:.2f} -- vr: {:.2f}".format(vLeft, vRight))

    error = sim.simxSetJointTargetVelocity(clientID, leftMotorHandle, vLeft, sim.simx_opmode_streaming)
    error = sim.simxSetJointTargetVelocity(clientID, rightMotorHandle, vRight, sim.simx_opmode_streaming)
#------------------------------------------------------------------------------#


#---------------------------- ESTABLISH CONNECTION ----------------------------#

import time
import sys

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim

if clientID!=-1:
    print ('Connected to remote API server')

    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res,objs=sim.simxGetObjects(clientID,sim.sim_handle_all,sim.simx_opmode_blocking)
    if res==sim.simx_return_ok:
        print ('Number of objects in the scene: ',len(objs))
    else:
        print ('Remote API function call returned with error code: ',res)

    time.sleep(2)
else:
    print ('Could not connect to server!')
    sys.exit()


print ('Server connected!')
#------------------------------------------------------------------------------#

#------------------------------- INITIALIZATION -------------------------------#
# motor initialization
leftMotorHandle = getHandle( 'Pioneer_p3dx_leftMotor' )
rightMotorHandle = getHandle( 'Pioneer_p3dx_rightMotor' )

# sonar initialization
sensorHandle = []
for i in range(16):
    sensorHandle.append( getHandle( "Pioneer_p3dx_ultrasonicSensor%d" % (i+1) ) )
    # set up streaming communication
    error, state, coord, detectedObjectHandle, detectedSurfaceNormalVector = sim.simxReadProximitySensor( clientID, sensorHandle[i],sim.simx_opmode_streaming )

# set up straming communication
sim.simxGetFloatSignal(clientID, "xPioneerPos", sim.simx_opmode_streaming)
sim.simxGetFloatSignal(clientID, "yPioneerPos", sim.simx_opmode_streaming)
sim.simxGetFloatSignal(clientID, "zPioneerPos", sim.simx_opmode_streaming)

sim.simxGetFloatSignal(clientID, "xFlagPos", sim.simx_opmode_streaming)
sim.simxGetFloatSignal(clientID, "yFlagPos", sim.simx_opmode_streaming)
sim.simxGetFloatSignal(clientID, "zFlagPos", sim.simx_opmode_streaming)

sim.simxGetFloatSignal(clientID, "compassAngle", sim.simx_opmode_streaming)
#------------------------------------------------------------------------------#

#--------------------------------- BUILD PATH ---------------------------------#

from buildPath import buildPath

sceneID = sys.argv[1]

errorFlagPos = errorPioneerPos = True

while( errorFlagPos or errorPioneerPos ):
    errorFlagPos, flagPos = readPositionSignal( "Flag" )
    errorPioneerPos, pioneerPos = readPositionSignal("Pioneer")

origin = np.array( [pioneerPos[0], pioneerPos[1]] )
destination = np.array( [flagPos[0], flagPos[1]] )

path = buildPath(sceneID, origin, destination, showPath=True)
point = 0


#------------------------------------------------------------------------------#

#---------------------------------- ACTUATION ---------------------------------#
verbose = False

while sim.simxGetConnectionId(clientID) != -1:

    vLeft, vRight, reachedFlag, pose, point = moveTowardsFlag( point, path, verbose=verbose )
    if( reachedFlag ):
        updateSpeed( vLeft, vRight, verbose=verbose )
        continue

    vLeft, vRight = followWall( vLeft, vRight, pose, "Point", verbose=verbose )
    # vLeft, vRight = braitenberg( vLeft, vRight, pose, "Point", verbose=verbose )

    # update motor speeds
    updateSpeed( vLeft, vRight, verbose=verbose )

# close connection
sim.simxFinish(clientID) 
print ('Closed connection!')
