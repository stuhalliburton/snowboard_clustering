import matplotlib.pyplot as plt

def plot_clusters(powder, park, all_mountain):
    plt.scatter(powder[0], powder[1], label='Powder', c='red')
    plt.scatter(park[0], park[1], label='Park', c='blue')
    plt.scatter(all_mountain[0], all_mountain[1], label='All Mountain', c='lightgreen')

    plt.legend()
    plt.show()
