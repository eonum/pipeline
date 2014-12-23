package ch.eonum.pipeline.clustering;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;
import ch.eonum.pipeline.core.PipelineReader;
import ch.eonum.pipeline.distance.Distance;
import ch.eonum.pipeline.distance.EuclidianDistance;
import ch.eonum.pipeline.util.Log;

/**
 * <p>Self Organizing Map / Kohonen Map Unsupervised 2-dimensional clustering. No
 * groundTruth needed. {@link http://en.wikipedia.org/wiki/Self-organizing_map}</p>
 * 
 * <p>The implemented map is a 2-dimensional network with a Gaussian neighborhood
 * function.</p>
 * 
 * <p>Testing/Classification Each test instance is assigned a network cell
 * (cluster), which will be encoded within the label field in the format "x-y",
 * where x and y denote the position of the cluster/network node. Additionally a
 * fuzzy value for the x- and y-axis are written into the features "x" and "y".
 * The fuzzy position is a weighted combination of all network nodes, according
 * the distance to the instance. They can be used to position an instance on a
 * map.</p>
 * 
 * @author tim
 * 
 */
public class SelfOrganizingMap<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();

	static {
		PARAMETERS.put("length",
						"network length in x and y direction (number of nodes), (default: 10)");
		PARAMETERS.put("learningRate",
				"learning rate or constant. (default: 0.1)");
		PARAMETERS.put("iterations", "number of iterations. (default: 1000)");
		PARAMETERS.put("radius", "initial radius. (default: 2.0)");
	}

	/** distance metric. */
	private Distance<Instance> distance;
	/** network nodes / weight vectors. */
	private SparseInstance[][] cells;

	public SelfOrganizingMap(Features features, Distance<Instance> distance) {
		super();
		this.setFeatures(features);
		this.setSupportedParameters(SelfOrganizingMap.PARAMETERS);
		this.putParameter("length", 10.0);
		this.putParameter("learningRate", 0.1);
		this.putParameter("iterations", 1000.0);
		this.putParameter("radius", 2.0);
		this.distance = distance;
	}

	public SelfOrganizingMap(Features features) {
		this(features, new EuclidianDistance<Instance>());
	}

	public SelfOrganizingMap() {
		this(null, new EuclidianDistance<Instance>());
	}

	@Override
	public void train() {
		int length = (int) this.getDoubleParameter("length");
		int numIterations = (int) this.getDoubleParameter("iterations");

		this.cells = new SparseInstance[length][length];
		/** random initialization of the weight vectors. */
		for (int x = 0; x < length; x++)
			for (int y = 0; y < length; y++) {
				cells[x][y] = new SparseInstance(x + "-" + y, "",
						new HashMap<String, Double>());
				for (int i = 0; i < this.features.size(); i++)
					cells[x][y].put(features.getFeatureByIndex(i),
							Math.random());
			}

		for (int iteration = 0; iteration < numIterations; iteration++) {
			double currentRadius = this
					.gaussianRadius(iteration, numIterations);
			double currentLR = this.expLR(iteration, numIterations);
			Log.puts("Iteration " + iteration + " (radius = " + currentRadius
					+ ", learning rate = " + currentLR + ")");

			for (E each : this.trainingDataSet) {
				/** select the nearest node. */
				double minDistance = Double.POSITIVE_INFINITY;
				int minx = 0;
				int miny = 0;
				for (int x = 0; x < length; x++)
					for (int y = 0; y < length; y++) {
						double distance = this.distance.distance(cells[x][y],
								each);
						if (distance < minDistance) {
							minDistance = distance;
							minx = x;
							miny = y;
						}
					}
				/** update the weight vectors. */
				for (int x = 0; x < length; x++)
					for (int y = 0; y < length; y++) {
						double weight = currentLR
								* this.neigborhoodFunction(x, y, minx, miny,
										currentRadius);
						for (String feature : cells[x][y].features())
							cells[x][y].put(
									feature,
									cells[x][y].get(feature)
											+ weight
											* (each.get(feature) - cells[x][y]
													.get(feature)));
					}
			}
		}

	}

	/**
	 * Neighborhood function currently this is a Gaussian neighborhood function.
	 * the function can be extended to support more functions (e.g. linear)
	 * 
	 * @param x1 x position of item 1
	 * @param y1 y position of item 1
	 * @param x2 x position of item 2
	 * @param y2 y position of item 2
	 * @param width width of the Gaussian
	 * @return
	 */
	private double neigborhoodFunction(double x1, double y1, double x2,
			double y2, double width) {
		SparseInstance inst1 = new SparseInstance("", "", new HashMap<String, Double>());
		inst1.put("x", x1);
		inst1.put("y", y1);
		SparseInstance inst2 = new SparseInstance("", "", new HashMap<String, Double>());
		inst2.put("x", x2);
		inst2.put("y", y2);
		double distance = this.distance.distance(inst1, inst2);
		return (Math.exp(-1.0 * distance * distance / (2.0 * width * width)));
	}

	/**
	 * Calculates the Gaussian neighborhood radius value.
	 * 
	 * @param int n - current step (time).
	 * @param int A - time constant (usually the number of iterations in the
	 *        learning process).
	 * @return double - adapted Gaussian neighborhood function value.
	 */
	private double gaussianRadius(int n, int A) {
		return (this.getDoubleParameter("radius") * Math.exp(-1.0
				* ((double) n) / ((double) A)));
	}

	/**
	 * Calculates the exponential learning-rate parameter value.
	 * 
	 * @param int n - current step (time).
	 * @param int A - time constant (usually the number of iterations in the
	 *        learning process).
	 * @return double - exponential learning-rate parameter value.
	 */
	private double expLR(int n, int A) {
		return (this.getDoubleParameter("learningRate") * Math.exp(-1.0
				* ((double) n) / ((double) A)));
	}

	/**
	 * assign a node and calculate x and y values.
	 */
	@Override
	public DataSet<E> test() {
		int length = this.cells.length;
		for (Instance each : this.testDataSet) {
			/** select the nearest node. */
			double minDistance = Double.POSITIVE_INFINITY;
			double maxDistance = Double.NEGATIVE_INFINITY;
			int minx = 0;
			int miny = 0;
			double[][] distanceCache = new double[length][length];
			for (int x = 0; x < length; x++)
				for (int y = 0; y < length; y++) {
					distanceCache[x][y] = this.distance.distance(cells[x][y],
							each);
					if (distanceCache[x][y] < minDistance) {
						minDistance = distanceCache[x][y];
						minx = x;
						miny = y;
					}
					if (distanceCache[x][y] > maxDistance)
						maxDistance = distanceCache[x][y];
				}
			each.label = minx + "-" + miny;
			/**
			 * calculate fuzzy x and y coordinates using a Gaussian neighborhood
			 * for the weights.
			 **/
			double xFuzzy = 0.0;
			double yFuzzy = 0.0;
			double totalWeight = 0.0;
			double maxDistance2 = 2.0 * maxDistance * maxDistance;
			for (int x = 0; x < length; x++)
				for (int y = 0; y < length; y++) {
					double weight = Math.exp(-10.0 * distanceCache[x][y]
							* distanceCache[x][y] / maxDistance2);
					totalWeight += weight;
					xFuzzy += weight * x;
					yFuzzy += weight * y;
				}
			xFuzzy /= totalWeight;
			yFuzzy /= totalWeight;
			each.putResult("x", xFuzzy);
			each.putResult("y", yFuzzy);
		}
		return this.testDataSet;
	}

	@Override
	public void save(String fileName) throws IOException {
		super.save(fileName);
		DataSet<SparseInstance> ds = new DataSet<SparseInstance>();
		for (int x = 0; x < cells.length; x++)
			for (int y = 0; y < cells[x].length; y++)
				ds.addInstance(cells[x][y]);
		ds.writeToFile(fileName + ".cells");
	}

	@Override
	public void loadSerializedState(File file) throws IOException {
		super.loadSerializedState(file);
		DataSet<SparseInstance> ds = new PipelineReader(file.getAbsolutePath() + ".cells")
				.readFromFile();
		int length = (int) this.getDoubleParameter("length");
		this.cells = new SparseInstance[length][length];
		for (SparseInstance each : ds) {
			int x = Integer.valueOf(each.id.split("-")[0]);
			int y = Integer.valueOf(each.id.split("-")[1]);
			this.cells[x][y] = each;
		}
	}

}
