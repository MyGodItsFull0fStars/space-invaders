import gym
import numpy as np
from matplotlib import pyplot as plt

from ipywidgets import widgets
from IPython.display import display

from matplotlib import animation
from JSAnimation.IPython_display import display_animation


def left_click():
    """
    Apply a force to the left of the cart
    """
    on_click(0)


def right_click():
    """
    Apply a force to the right of the cart
    """
    on_click(1)


def display_buttons():
    """
    Display the buttons you can use to apply a force to the cart
    """
    left = widgets.Button(description='<')
    right = widgets.Button(description='>')
    display(left, right)

    left.on_click(left_click)
    right.on_click(right_click)


def on_click(action: int):
    global frames
    observation, reward, done, info = env.step(action)
    frame = env.render(mode='rgb_array')
    im.set_data(frame)
    frames.append(frame)
    if done:
        env.reset()


def display_frames_as_gif(frames, filename_gif=None):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    if filename_gif:
        anim.save(filename_gif, writer='imagemagick', fps=20)
    display(display_animation(anim, default_mode='loop'))


def run_episode(env, parameters):
    """
    Runs the env for a certain amount of steps with the given parameters.
    :param env:
    :param parameters:
    :return: the reward obtained
    """
    observation = env.reset()
    total_reward: int = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def show_episode(env, parameters):
    """
    Records the frames of the environment obtained using the given parameters
    :param env:
    :param parameters:
    :return: RGB frames
    """
    observation = env.reset()
    first_frame = env.render(mode='rgb_array')
    frames = [first_frame]

    for _ in range(200):
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        if done:
            break
    return frames


def random_search():
    """
    Try random parameters between -1 and 1,
    see how long the game lasts with those parameters
    :return:
    """
    best_random_params = None
    best_random_reward = 0
    for _ in range(10000):
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        if reward > best_random_reward:
            best_random_reward = reward
            best_random_params = parameters
            # considered solved if the agent lasts 200 timesteps
            if reward == 200:
                break

    return best_random_reward, best_random_params


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation = env.reset()

    first_frame = env.render(mode='rgb_array')
    fig, ax = plt.subplots()

    im = ax.imshow(first_frame)

    best_reward, best_params = random_search()
    frames = show_episode(env, best_params)

    display_frames_as_gif(frames, filename_gif='best_result_random.gif')


    env.close()
