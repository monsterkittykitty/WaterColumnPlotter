import pstats

# p = pstats.Stats("profile-Capture.txt")
# p = pstats.Stats("profile-Plotter.txt")
p = pstats.Stats("profile-Process.txt")

# p.sort_stats('cumulative').print_stats(100)
p.sort_stats('tottime').print_stats(100)