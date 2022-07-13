import numpy.random as rd
import gudhi.representations
import numpy.random as rd

rd.seed(1)




normal_pcs = []
#uniform_pcs = []
for i in range(100):
    normal_pcs.append(rd.normal(size = [100,2]))
   # uniform_pcs.append(rd.uniform(size = [100,2], low=-2, high=2))

#we show the first point cloud of each type as an example
# fig, ax = plt.subplots()
# normal_pc = normal_pcs[0]
# uniform_pc = uniform_pcs[0]
# ax.scatter(normal_pc[:,0], normal_pc[:,1], s=15, label = 'normal')
# ax.scatter(uniform_pc[:,0], uniform_pc[:,1], s=15, label = 'uniform')
# ax.legend()
# plt.show()

# dgms_normal = []
# for pc in normal_pcs:
#     rips = gudhi.RipsComplex(points=pc).create_simplex_tree(max_dimension=1)
#     rips.compute_persistence()
#     dgms_normal.append(rips.persistence_intervals_in_dimension(0))

# dgms_uniform = []
# for pc in uniform_pcs:
#     rips = gudhi.RipsComplex(points=pc).create_simplex_tree(max_dimension=1)
#     rips.compute_persistence()
#     dgms_uniform.append(rips.persistence_intervals_in_dimension(0))

# gd.plot_persistence_barcode(dgms_normal[0])
# plt.show()

# we create a lambda function which removes the infinity bars from a barcode.
# remove_infinity = lambda barcode : np.array([bars for bars in barcode if bars[1]!= np.inf])
# # apply this operator to all barcodes.
# dgms_normal = list(map(remove_infinity, dgms_normal))
#dgms_uniform = list(map(remove_infinity, dgms_uniform))


#obtaining the entropy
# PE = gd.representations.Entropy()
# pe_normal = PE.fit_transform(dgms_normal)
#print(pe_normal[:9])

# pe_uniform = PE.fit_transform(dgms_uniform)
# pe_uniform[:9]

#differences using boxplots

#pe_normal_array = np.array(pe_normal[:,0])
#pe_uniform_array = np.array(pe_uniform[:,0])

#fig, ax = plt.subplots()

#Fix the boxplot
# bp = ax.boxplot([pe_normal_array, pe_uniform_array], labels=['Normal', 'Uniform'])
#
# #We change the axis letter size to see them better
# plt.rc('xtick', labelsize=20)
# plt.rc('ytick', labelsize=15)
#
# plt.show()


#using the entropy summary function in Gudhi
dgms_normal = []
for pc in normal_pcs:
    rips = gudhi.RipsComplex(points=pc).create_simplex_tree(max_dimension=2)
    rips.persistence()
    dgms_normal.append(rips.persistence_intervals_in_dimension(1))

# dgms_uniform = []
# for pc in uniform_pcs:
#     rips = gudhi.RipsComplex(points=pc).create_simplex_tree(max_dimension=2)
#     rips.persistence()
#     dgms_uniform.append(rips.persistence_intervals_in_dimension(1))

#An example of 1 dimensional barcode
gd.plot_persistence_barcode(dgms_normal[0])
#plt.show()

# Sample range is the interval where the Betti curve is calculated. It is an optional parameter
BC = gd.representations.BettiCurve(sample_range=[0,1.5], resolution = 150)
bc_normal = BC.fit_transform(dgms_normal)
#bc_uniform = BC.fit_transform(dgms_uniform)

# xseq = np.array(range(0,150))/100 #constructing the betti_curve for the normal case
# plt.plot(xseq, bc_normal[0])

#obtaining Entropy summary
ES = gd.representations.Entropy(mode='vector', sample_range=[0,1.5], resolution = 10, normalized = False)
es_normal = ES.fit_transform(dgms_normal)
print(es_normal)
#es_uniform = ES.fit_transform(dgms_uniform)

# xseq = np.array(range(0,151))/100
# plt.plot(xseq, es_normal[0])


###Note: The ES function is usually quite similar to the Betti curve although it can be completely different for others.###