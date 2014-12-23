package ch.eonum.pipeline.classification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.distance.Distance;
import ch.eonum.pipeline.distance.EuclidianDistance;
import ch.eonum.pipeline.util.Log;

/**
 * K-Nearest Neighbor regressor.
 * 
 * @author tim
 * 
 */
public class NearestNeighbor<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();

	static {
		PARAMETERS.put("k",
						"number of neighbors which are taken into account. (default: 3.0)");
		PARAMETERS.put("f",
						"density function for the neighbors (uniform|linear|gauss|inverse-distance) (default:uniform)");
	}

	protected Distance<E> distance;

	public NearestNeighbor() {
		this.setSupportedParameters(NearestNeighbor.PARAMETERS);
		this.putParameter("k", 3.0);
		this.putParameter("f", "uniform");
		this.distance = new EuclidianDistance<E>();
	}

	/**
	 * There is no training. The training data is the model.
	 */
	@Override
	public void train() {}

	@Override
	public DataSet<E> test() {
		int k = (int) this.getDoubleParameter("k");
		int num = 0;
		for (E inst : this.testDataSet) {
			num++;
			if (num % 100 == 0)
				Log.puts(num + " cases tested");
			List<Neighbor> neighbors = new ArrayList<Neighbor>();
			double length = this.distance.length(inst);
			for (E prototype : this.trainingDataSet) {
				if (inst == prototype)
					/**avoid overfitting when testing the training set. */
					continue;
				double distance = this.distance.distance(prototype, inst,
						length);
				neighbors.add(new Neighbor(prototype, distance));
			}
			Collections.sort(neighbors);
			double result = 0.0;
			double totalDensity = 0.0;
			for (int i = 0; i < k; i++) {
				double density = this.densityFunction(k, i, neighbors.get(i)
						.getDelta());
				totalDensity += density;
				result += density * neighbors.get(i).getInstance().outcome;
			}
			inst.putResult("result", result / totalDensity);
		}
		return this.testDataSet;
	}

	/**
	 * 
	 * @param k total number of neighbors
	 * @param i i'th nearest neighbor
	 * @param d distance of neighbor i to the test instance
	 * @return density - weight for neighbor i with distance d 
	 */
	protected double densityFunction(int k, int i, double d) {
		if ("uniform".equals(this.getStringParameter("f")))
			return 1;
		else if ("inverse-distance".equals(this.getStringParameter("f")))
			return 1 / d;
		else if ("linear".equals(this.getStringParameter("f")))
			return k - i;
		else if ("gauss".equals(this.getStringParameter("f"))) {
			double var2 = 2 * Math.pow((double) k / 3.00, 2);
			return (1 / Math.sqrt(Math.PI * var2)) * Math.exp(-(i * i) / var2);
		} else
			Log.error("Unsupported density function"
					+ this.getStringParameter("f"));
		return 0;
	}

	protected class Neighbor implements Comparable<Neighbor> {
		private E inst;
		private double delta;

		public Neighbor(E inst, double delta) {
			this.inst = inst;
			this.delta = delta;
		}

		public E getInstance() {
			return inst;
		}

		public double getDelta() {
			return delta;
		}

		@Override
		public int compareTo(Neighbor other) {
			return delta - other.getDelta() > 0 ? 1 : -1;
		}

		@Override
		public String toString() {
			return this.inst + ": " + this.delta;
		}
	}

	public void setDistanceFunction(Distance<E> distance) {
		this.distance = distance;
	}

}
