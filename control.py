import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    env.reset()

    Kp = 38.8
    Ki = 25.2
    Kd = 15

    force = 0
    integral = 0

    steps = [0.02*i for i in range(1000)]
    angles_list = []
    for _ in range(1000):
        env.render()

        observation, reward, done, _, info = env.step(force)

        angle = observation[2]
        angular_velocity = observation[3]

        integral = integral + angle

        F = Kp*angle + Kd*angular_velocity + Ki*integral

        force = 1 if F > 0 else 0
        if done:
            observation = env.reset()
            integral = 0

        angles_list.append(angle)

    env.close()
    
    plt.plot(steps, angles_list)
    plt.show()
