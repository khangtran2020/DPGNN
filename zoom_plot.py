fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.rcParams["figure.figsize"] = (10,7)
# plt.plot(distance_kdd, label="kdd", marker=next(marker))
# plt.plot(distance_01, label="p_rate = 0.1", marker=next(marker))
# plt.plot(distance_001, label="p_rate = 0.01",  marker=next(marker))
# plt.plot(distance_0001, label="p_rate = 0.001",  marker=next(marker))
# plt.plot(distance_02, label="p_rate = 0.2",  marker=next(marker))
# plt.plot(distance_05, label="p_rate = 0.5",  marker=next(marker))
# plt.plot(distance_075, label="p_rate = 0.75",  marker=next(marker))
# plt.plot(distance_09, label="p_rate = 0.9",  marker=next(marker))
# plt.plot(distance_1, label="p_rate = 1",  marker=next(marker))


# plot big picture here 


plt.title("RMS distance from original graph with generated graph\n with random walk")
plt.legend()
plt.ylabel("RMS distance")
plt.xlabel("hops")

ax_new = fig.add_axes([0.4, 0.4, 0.3, 0.3]) # position of the zoom

# plt.plot(distance_kdd, label="kdd", marker=next(marker))
# plt.plot(distance_01, label="p_rate = 0.1", marker=next(marker))
# plt.plot(distance_001, label="p_rate = 0.01",  marker=next(marker))
# plt.plot(distance_0001, label="p_rate = 0.001",  marker=next(marker))
# plt.plot(distance_02, label="p_rate = 0.2",  marker=next(marker))
# plt.plot(distance_05, label="p_rate = 0.5",  marker=next(marker))
# plt.plot(distance_075, label="p_rate = 0.75",  marker=next(marker))
# plt.plot(distance_09, label="p_rate = 0.9",  marker=next(marker))
# plt.plot(distance_1, label="p_rate = 1",  marker=next(marker))

# plot zoom picture here 

plt.ylim(bottom = 34, top = 36)
plt.xlim(left = 4, right = 10)