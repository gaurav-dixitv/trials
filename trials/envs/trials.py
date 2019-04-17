import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef,
                      polygonShape, revoluteJointDef, contactListener,
                      prismaticJointDef, distanceJointDef, wheelJointDef)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from gym.envs.classic_control import rendering

import pyglet
from pyglet.window import key
import pyglet.gl as gl


# seaborn colors
colors = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
          (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
          (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
          (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
          (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
          (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
          (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
          (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
          (0.8, 0.7254901960784313, 0.4549019607843137),
          (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]


class BikeDef:
    """Defines constants for bike geometry and joints
    """

    # fixture151
    chassis_poly_2 = [
        (0.5925927162170410, 0.3970571756362915),
        (0.5049575567245483, 0.5745437145233154),
        (-0.1241288781166077, 0.3671160936355591),
        (-0.3777002990245819, 0.01054022461175919),
        (-0.1886725425720215, -0.3709129691123962),
        (-0.01139009092003107, -0.4074897468090057),
        (0.2494194805622101, -0.3823029100894928),
        (0.4150831103324890, -0.01518678665161133)
    ]
    chassis_fd_2 = fixtureDef(
        shape=polygonShape(
            vertices=[(x, y) for x, y in chassis_poly_2]),
        density=1.0,
        friction=0.1,
        categoryBits=0x0020,
        restitution=0.1)

    # fixture155
    chassis_poly_3 = [
        (-0.1139258146286011, 0.3664703369140625),
        (-0.7596390247344971, 0.4362251460552216),
        (-0.7564918398857117, 0.3531839251518250),
        (-0.3745602667331696, 0.004400946199893951)
    ]
    chassis_fd_3 = fixtureDef(
        shape=polygonShape(
            vertices=[(x, y) for x, y in chassis_poly_3]),
        density=0.01999,
        friction=0.10,
        categoryBits=0x0020,
        restitution=0.10)

    # fixture153
    chassis_poly_4 = [
        (0.9036020040512085, 0.4347184896469116),
        (0.5974624752998352, 0.4441443085670471),
        (0.4807239770889282, 0.2040471434593201),
        (0.8664048910140991, 0.3509610295295715)
    ]
    chassis_fd_4 = fixtureDef(
        shape=polygonShape(
            vertices=[(x, y) for x, y in chassis_poly_4]),
        density=3.0,
        friction=0.1,
        categoryBits=0x0020,
        restitution=0.1)

    # fixture152
    chassis_poly_5 = [
        (0.8914304971694946, -0.07865893840789795),
        (0.4427454471588135, 0.8150310516357422),
        (0.3581824004650116, 0.7722809314727783),
        (0.8068674802780151, -0.1214089989662170)
    ]
    chassis_fd_5 = fixtureDef(
        shape=polygonShape(
            vertices=[(x, y) for x, y in chassis_poly_5]),
        density=3.0,
        friction=0.1,
        categoryBits=0x0020,
        restitution=0.1)

    # fixture156
    chassis_poly_6 = [
        (-0.7481837272644043, 0.4318268001079559),
        (-1.322828769683838, 0.4704772233963013),
        (-1.085322141647339, 0.3916769027709961),
        (-0.7573439478874207, 0.3553154766559601)
    ]
    chassis_fd_6 = fixtureDef(
        shape=polygonShape(
            vertices=[(x, y) for x, y in chassis_poly_6]),
        density=1.0,
        friction=0.1,
        categoryBits=0x0020,
        maskBits=0x00,
        restitution=0.1)

    # fixture150
    chassis_poly_7 = [
        (0.5167351365089417, 0.8024124503135681),
        (0.258435457944870, 0.821202814579010),
        (0.2579171955585480, 0.7468612790107727),
        (0.5162168145179749, 0.7280706763267517)
    ]
    chassis_fd_7 = fixtureDef(
        shape=polygonShape(
            vertices=[(x, y) for x, y in chassis_poly_7]),
        density=1.0,
        friction=0.1,
        categoryBits=0x0020,
        restitution=0.1)

    # fixture154
    chassis_poly_8 = [
        (1.313481688499451, 0.2247920632362366),
        (1.158682703971863, 0.3579198122024536),
        (0.9046562314033508, 0.4359158873558044),
        (0.8647161722183228, 0.3494157195091248)
    ]
    chassis_fd_8 = fixtureDef(
        shape=polygonShape(
            vertices=[(x, y) for x, y in chassis_poly_8]),
        density=3.0,
        friction=0.1,
        categoryBits=0x0020,
        restitution=0.1)

    # fixture11
    chassis_poly_9 = [
        (-0.4820806682109833, -0.3676601052284241),
        (-0.5027866363525391, -0.2698271274566650),
        (-0.6006187200546265, -0.2905341386795044),
        (-0.5799126625061035, -0.3883671164512634)
    ]
    chassis_fd_9 = fixtureDef(
        shape=polygonShape(
            vertices=[(x, y) for x, y in chassis_poly_9]),
        density=76.0,
        friction=0.2,
        categoryBits=0x0020,
        maskBits=0x00,
        restitution=0.0)

    # fixture157
    front_wheel_fd = fixtureDef(
        shape=circleShape(radius=0.492374),
        density=0.07999,
        restitution=0.200,
        friction=0.0100,
        categoryBits=0x0020,
    )

    # fixture159
    rear_wheel_fd = fixtureDef(
        shape=circleShape(radius=0.450131),
        density=0.01,
        restitution=0.100,
        friction=0.600,
        categoryBits=0x0020
    )

    # fixture158
    swingarm_poly = [
        (0.4536499977111816, 0.06558109819889069),
        (0.4325839877128601, 0.1579640060663223),
        (-0.4387759864330292, -0.04519569873809814),
        (-0.4177089929580688, -0.1375789940357208)
    ]
    swingarm_fd = fixtureDef(
        shape=polygonShape(
            vertices=[(x, y) for x, y in swingarm_poly]),
        density=0.5000,
        restitution=0.100,
        friction=0.100,
        categoryBits=0x0020,
    )


class Bike:
    """Create/Reset/Destroy Bikes.

    """

    def __init__(self, world: Box2D.b2World, init_x: float, init_y: float):
        """create a new bike at init_x, init_y in world

        Arguments:
            world {Box2D.b2World} -- World to create the bike in.
            init_x {float} -- Initial x coodinate of the bike in the world.
            init_y {float} -- Initial y coodinate of the bike in the world.
        """

        self.world = world
        self._create(init_x, init_y)

    def parts(self):
        """Bike parts that use physics.

        Returns:
            [Box2D.b2_dynamicBody] -- parts that use physics.
        """

        return [self.chassis, self.front_wheel, self.rear_wheel, self.swingArm]

    def _create(self, init_x: float, init_y: float):
        """Internal method to create a bike at init_x, init_y in self.world

        Arguments:
            init_x {float} -- Initial x coodinate of the bike in the world.
            init_y {float} -- Initial y coodinate of the bike in the world.
        """

        self.chassis = self.world.CreateDynamicBody(
            position=(init_x + 15.45116233825684,
                      init_y + -0.1342248022556305),
            fixtures=(BikeDef.chassis_fd_2, BikeDef.chassis_fd_3,
                      BikeDef.chassis_fd_4, BikeDef.chassis_fd_5,
                      BikeDef.chassis_fd_6, BikeDef.chassis_fd_7,
                      BikeDef.chassis_fd_8, BikeDef.chassis_fd_9),
        )
        self.chassis.userData = "chassis"
        self.chassis.inertia = 0.8145936131477356
        self.chassis.mass = 1.939267754554749
        self.chassis.localCenter = (0.008193328045308590, 0.002193222753703594)

        self.chassis.color1 = colors[2]
        self.chassis.color2 = colors[9]

        self.front_wheel = self.world.CreateDynamicBody(
            position=(init_x + 16.49472427368164,
                      init_y + -0.5031779408454895),
            fixtures=BikeDef.front_wheel_fd
        )
        self.front_wheel.userData = "front_wheel"
        self.front_wheel.inertia = 0.00738
        self.front_wheel.mass = 0.0609
        self.front_wheel.localCenter = (
            0.008193328045308590, 0.002193222753703594)
        self.front_wheel.color1 = colors[7]
        self.front_wheel.color2 = colors[7]

        self.rear_wheel = self.world.CreateDynamicBody(
            position=(init_x + 14.52096557617188,
                      init_y + -0.5429208278656006),
            fixtures=BikeDef.rear_wheel_fd
        )
        self.rear_wheel.userData = "rear_wheel"
        self.rear_wheel.inertia = 0.00997
        self.rear_wheel.mass = 0.07993
        self.rear_wheel.localCenter = (
            -6.917995953870104e-09, 1.237957203414908e-08)
        self.rear_wheel.color1 = colors[7]
        self.rear_wheel.color2 = colors[7]

        self.swingArm = self.world.CreateDynamicBody(
            position=(init_x + 14.89060020446777,
                      init_y + -0.4709600210189819),
            fixtures=BikeDef.swingarm_fd
        )
        self.swingArm.userData = "swing_arm"
        self.swingArm.inertia = 0.0028663347475
        self.swingArm.mass = 0.04238938167
        self.swingArm.localCenter = (0.00743678025901, 0.0101925022900)
        self.swingArm.color1 = colors[7]
        self.swingArm.color2 = colors[7]

        self._createJoints()

    def _createJoints(self):
        """Internal method to create joints.
        """

        front_wheel_joint = wheelJointDef(
            bodyA=self.chassis,
            bodyB=self.front_wheel,
            localAnchorA=(1.096710562705994, -0.4867535233497620),
            localAnchorB=(-0.001052246429026127, 0.0007888938998803496),
            localAxisA=(-0.4521348178386688, 0.8919496536254883),
            frequencyHz=10.0,
            enableMotor=True,
            maxMotorTorque=20.0,
            dampingRatio=0.8
        )

        swingarm_revolute = revoluteJointDef(
            bodyA=self.chassis,
            bodyB=self.swingArm,
            localAnchorA=(-0.1697492003440857, -0.2360641807317734),
            localAnchorB=(0.3914504945278168, 0.1011910215020180),
            enableMotor=False,
            enableLimit=True,
            maxMotorTorque=9999999999.0,
            motorSpeed=0,
            lowerAngle=0.1,
            upperAngle=0.45
        )

        # distancejoint4
        swingarm_distance = distanceJointDef(
            bodyA=self.chassis,
            bodyB=self.swingArm,
            localAnchorA=(-0.8274764418601990, 0.3798926770687103),
            localAnchorB=(-0.2273961603641510, -0.04680897668004036),
            length=0.8700000047683716,
            frequencyHz=8.0,
            dampingRatio=0.8999999761581421
        )

        # drivejoint
        swingarm_drive = revoluteJointDef(
            bodyA=self.swingArm,
            bodyB=self.rear_wheel,
            localAnchorA=(-0.3686238229274750, -0.07278718799352646),
            localAnchorB=(0, 0),
            enableMotor=False,
            enableLimit=False,
            maxMotorTorque=8.0,
            motorSpeed=0,
            lowerAngle=0,
            upperAngle=0,
        )

        self.front_wheel_joint = self.world.CreateJoint(front_wheel_joint)
        self.swingarm_revolute = self.world.CreateJoint(swingarm_revolute)
        self.swingarm_distance = self.world.CreateJoint(swingarm_distance)
        self.swingarm_drive = self.world.CreateJoint(swingarm_drive)

    def _destroy(self):
        """Internal method to destroy bike bodies and joints.
        """

        if self.chassis:

            # destroy joints
            self.world.DestroyJoint(self.front_wheel_joint)
            self.world.DestroyJoint(self.swingarm_revolute)
            self.world.DestroyJoint(self.swingarm_distance)
            self.world.DestroyJoint(self.swingarm_drive)

            # destroy bodies
            self.world.DestroyBody(self.chassis)
            self.world.DestroyBody(self.front_wheel)
            self.world.DestroyBody(self.rear_wheel)
            self.world.DestroyBody(self.swingArm)

            self.chassis = None
            self.front_wheel = None
            self.rear_wheel = None
            self.swingArm = None

            self.front_wheel_joint = None
            self.swingarm_revolute = None
            self.swingarm_distance = None
            self.swingarm_drive = None

    def reset(self, init_x: float, init_y: float):
        """Destroy and re-create the bike at init_x, init_y in self.world

        Arguments:
            init_x {float} -- Initial x coodinate of the bike in the world.
            init_y {float} -- Initial y coodinate of the bike in the world.
        """

        self._destroy()
        self._create(init_x, init_y)


class LidarCallback(Box2D.b2.rayCastCallback):

    def ReportFixture(self, fixture: Box2D.b2Fixture, point: Box2D.b2Vec2, normal: Box2D.b2Vec2, fraction: float):
        """Raycast callback

        Arguments:
            fixture {Box2D.b2Fixture} -- the b2Fixture that was hit by the ray
            point {Box2D.b2Vec2} -- point of intersection
            normal {Box2D.b2Vec2} -- the normal vector at the point of intersection
            fraction {float} -- the fraction len(ray_after_intersection)/len(ray)

        Returns:
            {int} -- [check b2RayCastCallback for valid values to return]
        """

        if (fixture.filterData.categoryBits & 1) == 0:
            return 1
        self.p2 = point
        self.fraction = fraction
        return 0


class TrialsContactListener(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact: Box2D.b2Contact):
        """Callback when two fixtures in the world begin to touch. 
        Update bike state: wheels on ground and chassis state

        Arguments:
            contact {b2Contact} -- contact data
        """

        if contact.fixtureA.body.userData.startswith("terrain") or contact.fixtureB.body.userData.startswith("terrain"):
            if self.env.bike.chassis == contact.fixtureA.body or self.env.bike.chassis == contact.fixtureB.body:
                self.env.bike_body_on_ground = True

            if self.env.bike.front_wheel == contact.fixtureA.body or self.env.bike.front_wheel == contact.fixtureB.body:
                self.env.front_wheel_on_ground += 1

            if self.env.bike.rear_wheel == contact.fixtureA.body or self.env.bike.rear_wheel == contact.fixtureB.body:
                self.env.rear_wheel_on_ground += 1

    def EndContact(self, contact: Box2D.b2Contact):
        """Callback when two fixtures in the world end touch.
        Update bike state: wheels on ground and chassis state

        Arguments:
            contact {Box2D.b2Contact} -- [description]
        """

        if contact.fixtureA.body.userData.startswith("terrain") or contact.fixtureB.body.userData.startswith("terrain"):
            if self.env.bike.chassis == contact.fixtureA.body or self.env.bike.chassis == contact.fixtureB.body:
                self.env.bike_body_on_ground = False

            if self.env.bike.front_wheel == contact.fixtureA.body or self.env.bike.front_wheel == contact.fixtureB.body:
                self.env.front_wheel_on_ground -= 1

            if self.env.bike.rear_wheel == contact.fixtureA.body or self.env.bike.rear_wheel == contact.fixtureB.body:
                self.env.rear_wheel_on_ground -= 1


class TrialsViewer(rendering.Viewer):
    """Hack: Anti-aliasing.
    """

    def __init__(self, width, height):
        super().__init__(width, height)
        self.window.close()
        self.window = pyglet.window.Window(
            width=width, height=height, config=gl.Config(sample_buffers=1, samples=4))
        self.window.on_close = self.window_closed_by_user
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


def set_color(self, r: float, g: float, b: float, a=1.0):
    """Monkey patch to enable alpha
    Arguments:
        r {float} -- red [0.0, 1.0]
        g {float} -- green [0.0, 1.0]
        b {float} -- blue [0.0, 1.0]

    Keyword Arguments:
        a {float} -- alpha (default: {1.0})
    """
    self._color.vec4 = (r, g, b, a)


rendering.Geom.set_color = set_color  # Monkey_patch to allow alpha


class Trials(gym.Env, EzPickle):

    fps = 60
    generate_artifacts = False

    primary_width = 1024.0
    primary_height = 768.0
    screen = pyglet.window.get_platform().get_default_display().get_default_screen()
    viewport_width = int(screen.width * 0.5)
    viewport_height = int((primary_height/primary_width) * viewport_width)

    camera_width = primary_width * 0.03
    camera_height = primary_height * 0.03

    terrain_step = 2
    terrain_length = 300    # in steps
    terrain_height = 500
    terrain_startpad = 10    # in steps

    init_x = -12
    init_y = terrain_height + 1

    lidar_range = primary_height  # The max length of the casted ray
    lidar_count = 20

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': fps
    }

    def __init__(self, human_control=False, debug=False):
        EzPickle.__init__(self)
        self.seed()

        self.human_control = human_control
        self.debug = debug

        self.viewer = TrialsViewer(
            self.viewport_width, self.viewport_height)

        self.world = Box2D.b2World()
        self.world.continuousPhysics = False  # seems to affect joint limits
        self.terrain = None

        self.bike = Bike(self.world, self.init_x, self.init_y)

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0),
                                      (1, 1)]),
            friction=0.8,
            categoryBits=0x0001,
        )

        if self.human_control:
            self.move = [False for i in range(4)]
            self.viewer.window.on_key_press = self.key_press
            self.viewer.window.on_key_release = self.key_release

        self._resets = 0
        self.reset()

        high = np.array([np.inf] * (self.lidar_count + 5))
        self.action_space = spaces.Box(
            np.array([-1, -1, -1]), np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k == key.LEFT:
            self.move[0] = True
        if k == key.RIGHT:
            self.move[1] = True
        if k == key.UP:
            self.move[2] = True
        if k == key.DOWN:
            self.move[3] = True

    def key_release(self, k, mod):
        if k == key.LEFT:
            self.move[0] = False
        if k == key.RIGHT:
            self.move[1] = False
        if k == key.UP:
            self.move[2] = False
        if k == key.DOWN:
            self.move[3] = False
        if k == key.SPACE:
            self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.legs = []
        self.joints = []
        self.lidar = []

    def _generate_terrain(self):

        velocity = 0.0
        y = self.terrain_height
        self.terrain_x = []
        self.terrain_y = []
        amplitude = 0.5

        artifact_locations = np.random.random_integers(
            self.terrain_startpad + 10,
            self.terrain_length - 6.0 * 2.0,
            np.random.randint(3, 6)
        )

        def lerp(a, b, t): return a * (1 - t) + b * t
        i = 0
        while i < self.terrain_length:
            x = i*self.terrain_step
            self.terrain_x.append(x)
            velocity = lerp(0.7, 0.9, i/self.terrain_length) * \
                velocity + 0.01*np.sign(self.terrain_height - y)
            if i > self.terrain_startpad:
                velocity += self.np_random.uniform(-amplitude,
                                                   amplitude)
            y += velocity
            self.terrain_y.append(y)
            i += 1

            if i in artifact_locations and self.generate_artifacts:
                artifact_type = np.random.randint(1, 4)
                if artifact_type == 1:
                    # ramps
                    ramp_width = np.random.randint(3, 9)
                    ramp_height = np.random.randint(1, 4)
                    dy = y + ramp_height * np.sign(self.terrain_height - y)
                    for j in range(0, ramp_width):
                        x = i*self.terrain_step
                        i += 1
                        dy += 0.02*velocity + 0.001 * \
                            np.sign(self.terrain_height - dy)
                        self.terrain_x.append(x)
                        self.terrain_y.append(dy)
                elif artifact_type == 2:
                    # stairs
                    dy = y
                    sign = np.sign(self.terrain_height - y)
                    stair_len = np.random.randint(2, 4)
                    stair_count = np.random.randint(3, 6)
                    for j in range(0, stair_count):
                        dy += 2 * sign
                        for k in range(0, stair_len):
                            x = i*self.terrain_step
                            self.terrain_x.append(x)
                            self.terrain_y.append(dy)
                            i += 1
                    if sign < 0.0 or np.random.rand() > 0.5:
                        y = dy
                else:
                    # ninja 1
                    pit_height = np.random.randint(4, 6)
                    platform_count = np.random.randint(3, 4)
                    platform_width = np.random.randint(2, 4)
                    for j in range(platform_count):
                        dy = y - pit_height
                        x = (i-1)*self.terrain_step
                        self.terrain_x.append(x)
                        self.terrain_y.append(dy)
                        i += 1

                        x = i*self.terrain_step
                        self.terrain_x.append(x)
                        self.terrain_y.append(dy)

                        dy = y
                        for k in range(platform_width):
                            x = i*self.terrain_step
                            self.terrain_x.append(x)
                            self.terrain_y.append(dy)
                            i += 1

                    i += platform_count

        self._create_driving_line()

    def _linear_driving_line(self):

        self.terrain_x = []
        self.terrain_y = []
        for i in range(self.terrain_length):
            x = i*self.terrain_step
            self.terrain_x.append(x)
            y = self.terrain_height + 0
            self.terrain_y.append(y)

        self._create_driving_line()

    def _create_driving_line(self):
        self.terrain = []
        self.terrain_poly = []
        for i in range(self.terrain_length-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge)
            t.userData = "terrain"+str(i)
            color = (0.8, 0.6, 0.2, 0.6)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.9, 0.6, 0.3)
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _create_trees(self):
        self.trees = []

        tree_count = np.random.randint(15, 20)

        def sample(a, b): return (b - a) * np.random.random_sample() + a

        for tc in range(tree_count):
            index = np.random.randint(3, len(self.terrain_x)-3)
            base_x = self.terrain_x[index]
            base_y = self.terrain_y[index]

            greens = []
            tree_type = np.random.randint(1, 3)
            bark_color = np.random.choice([1, 5, 8])
            greens_color = np.random.choice([2, 5, 6, 7])
            if tree_type == 1:
                trunk_width = sample(0.1, 0.3)
                trunk_height = sample(0.5, 4.0)
                trunk = [(base_x-trunk_width, base_y),
                         (base_x-trunk_width, base_y+trunk_height),
                         (base_x+trunk_width, base_y+trunk_height),
                         (base_x+trunk_width, base_y)]

                green_width = sample(1.0, 2.0)
                green_height = sample(3.5, 7.0)
                greens_count = np.random.randint(1, 4)
                shoot_inc = 0
                for gc in range(greens_count):
                    greens.append(
                        (base_x-green_width, base_y+trunk_height + shoot_inc))
                    greens.append(
                        (base_x, base_y+trunk_height+green_height + shoot_inc))
                    greens.append(
                        (base_x+green_width, base_y+trunk_height + shoot_inc))
                    shoot_inc += np.random.randint(1, 3)

            else:
                trunk_width = sample(0.2, 0.3)
                trunk_height = sample(10.5, 18.0)
                trunk = [(base_x-trunk_width, base_y),
                         (base_x-trunk_width, base_y+trunk_height),
                         (base_x+trunk_width, base_y+trunk_height),
                         (base_x+trunk_width, base_y)]

                green_width = sample(1.0, 2.0)
                green_height = sample(1.5, 2.5)
                greens_count = np.random.randint(20, 30)
                shoot_inc = 0  # random increment
                shoot_dis = 0  # random displacement
                trunk_height = trunk_height * sample(0.5, 0.7)
                for gc in range(greens_count):
                    greens.append((base_x+shoot_dis-green_width,
                                   base_y+trunk_height + shoot_inc))
                    greens.append((base_x+shoot_dis, base_y +
                                   trunk_height+green_height + shoot_inc))
                    greens.append((base_x+shoot_dis+green_width,
                                   base_y+trunk_height + shoot_inc))
                    shoot_inc += np.random.randint(0.5, 2)
                    shoot_dis = np.random.randint(-1, 1)

            self.trees.append([trunk, (*colors[bark_color], 0.3),
                               tree_type, greens, (*colors[greens_color], 0.3)])

    def reset(self):
        self._destroy()
        self.world.contactListener = TrialsContactListener(self)
        self.game_over = False
        self.scroll = 0.0
        self.lidar_render = 0

        self.front_wheel_on_ground = 0
        self.rear_wheel_on_ground = 0
        self.bike_body_on_ground = False

        self._generate_terrain()
        self._create_trees()
        self.bike.reset(self.init_x, self.init_y)
        self.lidar = [LidarCallback() for _ in range(self.lidar_count)]

        self.last_pos = self.bike.chassis.position[0]

        return self.step(np.array([0, 0, 0, 0]))[0]

    def step(self, action):

        acc_velocity = 200
        flip_velocity = 1.25

        if self.human_control:
            if self.move[0]:
                self.bike.chassis.angularVelocity = flip_velocity
            if self.move[1]:
                self.bike.chassis.angularVelocity = -flip_velocity
            if self.move[2]:
                self.bike.rear_wheel.angularVelocity = -acc_velocity
            if self.move[3]:
                self.bike.rear_wheel.angularVelocity = acc_velocity

        if action is not None:
            if np.sign(action[0]) >= 0:
                self.bike.chassis.angularVelocity = float(
                    flip_velocity * np.sign(action[1]))
                self.bike.rear_wheel.angularVelocity = float(
                    -acc_velocity * np.sign(action[2]))

        self.world.Step(1.0/self.fps, 8 * 10, 3 * 10)

        pos = self.bike.chassis.position
        lidar_pos = list(pos)
        lidar_pos[1] += 2  # adjust for driver head.
        velocity = self.bike.chassis.linearVelocity
        speed = velocity.length

        low_angle = -75.0
        high_angle = 80.0
        angle_step = (high_angle - low_angle)/(self.lidar_count - 1)

        for i in range(self.lidar_count):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = lidar_pos
            angle = np.deg2rad(low_angle + angle_step * i)
            self.lidar[i].p2 = (
                lidar_pos[0] + math.sin(angle)*self.lidar_range,
                lidar_pos[1] - math.cos(angle)*self.lidar_range)
            self.world.RayCast(
                self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        track_length = self.terrain_length * self.terrain_step
        reward = 0
        if pos[0] > self.last_pos:
            self.last_pos = pos[0]
            reward = (pos[0] / track_length) * 10 + speed * 2.0
        else:
            reward = -1

        crashed = False
        if self.bike_body_on_ground:
            if abs(self.bike.chassis.angle) > np.deg2rad(150):
                self.game_over = True
                crashed = True

        done = False
        if self.game_over or pos[0] >= track_length or pos[0] < 0:
            done = True
            if pos[0] < 0:
                reward = -500
            elif pos[0] >= track_length:
                reward = 500
            elif crashed:
                reward = -800
            else:
                raise RuntimeWarning("Invalid bike state.")

        state = [
            1.0 if self.front_wheel_on_ground > 0 else 0.0,
            1.0 if self.rear_wheel_on_ground > 0 else 0.0,
            self.bike.chassis.angle,
            velocity[0],
            velocity[1]
        ]
        state += [l.fraction for l in self.lidar]

        self.scroll = pos[0] - self.viewport_width/3
        return state, reward, done, {"track_covered_percent": (pos[0] / track_length)}

    def render(self, mode='human'):

        pos = self.bike.chassis.position
        # The bike needs to be near the left edge and bottom of the camera
        self.viewer.set_bounds(pos[0] - self.camera_width * 0.5, pos[0] +
                               self.camera_width * 1.5, pos[1] - self.camera_height * 0.8, pos[1] + self.camera_height * 1.2)

        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + self.viewport_width:
                continue
            self.viewer.draw_polygon(
                poly, color=(*colors[5], 0.4), filled=True)
            self.viewer.draw_polyline(
                [poly[0], poly[1]], color=colors[5], linewidth=1.2)

        self.lidar_render = (self.lidar_render+1) % 100
        i = self.lidar_render
        if i < 2*len(self.lidar):
            l = self.lidar[i] if i < len(
                self.lidar) else self.lidar[len(self.lidar)-i-1]
            self.viewer.draw_polyline(
                [l.p1, l.p2], color=(*colors[3], 0.3), linewidth=2)

        for tree in self.trees:
            self.viewer.draw_polygon(
                tree[0],
                color=tree[1],
                filled=True
            )
            tree_type = tree[2]
            if tree_type == 1:
                self.viewer.draw_polygon(
                    tree[3],
                    color=tree[4],
                    filled=True
                )
            else:
                greens = tree[3]
                green = 0
                while green < len(greens):
                    self.viewer.draw_polygon(
                        greens[green:green+3],
                        color=tree[4],
                        filled=True
                    )
                    green += 3

        for obj in self.bike.parts():
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(
                        f.shape.radius, 30, color=(*obj.color1, 0.1), filled=True).add_attr(t)
                    self.viewer.draw_circle(
                        f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(
                        path, color=(*obj.color1, 0.3), filled=True)
                    path.append(path[0])
                    self.viewer.draw_polyline(
                        path, color=obj.color2, linewidth=1.8)

        if self.debug:
            for joint_edge in self.bike.chassis.joints:
                joint = joint_edge.joint
                anchorA = joint.anchorA
                anchorB = joint.anchorB
                self.viewer.draw_polyline(
                    [(anchorA[0], anchorA[1]),
                     (anchorB[0], anchorB[1])],
                    color=colors[0],
                    linewidth=2
                )

        flagy_y1 = self.terrain_y[-1]
        flagy_y2 = flagy_y1 + 1.5
        x = self.terrain_x[-1]
        self.viewer.draw_polyline(
            [(x, flagy_y1), (x, flagy_y2)], color=(0, 0, 0), linewidth=1)
        f = [(x, flagy_y2), (x, flagy_y2-0.5), (x+0.7, flagy_y2-0.25)]
        self.viewer.draw_polygon(f, color=(*colors[2], 0.3))
        self.viewer.draw_polyline(
            f + [f[0]], color=(*colors[2], 0.9), linewidth=2)

        if self.human_control:
            self._render_keys(pos)  #TODO render agent actions.

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _render_keys(self, pos):

        k_w = 1.0
        k_h = 1.0
        layout = 'arcade' # the other is 'zen'.
        if layout == 'arcade':
            padding = 0.5
            u_x = pos[0] - k_w * 6.0
            u_y = pos[1] - 5
            d_x = u_x
            d_y = u_y - k_h - padding
            l_x = u_x - padding * 0.5
            l_y = u_y - padding * 0.5
            r_x = u_x + padding * 0.5 + k_w * 1.0
            r_y = l_y
        else:
            padding = 2.0
            u_x = pos[0] - k_w * 4.0 * 0.5
            u_y = pos[1] - 5
            d_x = u_x + k_w * padding
            d_y = u_y
            l_x = d_x + k_w * padding
            l_y = d_y
            r_x = l_x + k_w * padding * 0.5
            r_y = l_y

        self.viewer.draw_polygon(
            [(u_x, u_y), (u_x + k_w * 0.5, u_y + k_h * 0.5), (u_x + k_w, u_y)],
            color=colors[7],
            filled=self.move[2]
        )
        self.viewer.draw_polygon(
            [(d_x, d_y), (d_x + k_w * 0.5, d_y - k_h * 0.5), (d_x + k_w, d_y)],
            color=colors[7],
            filled=self.move[3]
        )
        self.viewer.draw_polygon(
            [(l_x, l_y), (l_x - k_w * 0.5, l_y - k_h * 0.5), (l_x, l_y - k_h)],
            color=colors[7],
            filled=self.move[0]
        )
        self.viewer.draw_polygon(
            [(r_x, r_y), (r_x + k_w * 0.5, r_y - k_h * 0.5), (r_x, r_y - k_h)],
            color=colors[7],
            filled=self.move[1]
        )

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# human play.
if __name__ == "__main__":

    env = Trials(human_control=True, debug=False)
    env.reset()
    while True:
        obs, reward, done, info = env.step(None)
        env.render()
        if done:
            env.reset()
