package ch.eonum.pipeline.distance;

import ch.eonum.pipeline.core.DenseSequence;
import ch.eonum.pipeline.core.Features;

/**
 * Dynamic Time Warping. @see http://en.wikipedia.org/wiki/Dynamic_time_warping.
 * Measures the similarity/distance between two temporal sequences by aligning
 * them using dynamic programming. Currently this is a simple dynamic time
 * warping distance without any optimization. Setting different weights for
 * different features is done by scaling those features.
 * 
 * Only for dense sequences. Cost function is a simple euclidian distance.
 * 
 * @author tim
 * 
 * @param <E>
 */
public class DynamicTimeWarpingDense<E extends DenseSequence> extends Distance<E> {

	/** Feature set. */
	private Features features;

	/**
	 * 
	 * @param zi
	 *            zero instance
	 * @param costFunction
	 *            Cost function to measure the transformation cost from one
	 *            point in a sequence to another point in a sequence.
	 */
	public DynamicTimeWarpingDense(E zi, Features features) {
		super(zi);
		this.features = features;
	}

	@Override
	public double distance(E s, E t) {
		int n = s.getSequenceLength() + 1;
		int m = t.getSequenceLength() + 1;
		double[][] sArray = s.getDenseRepresentation(features);
		double[][] tArray = t.getDenseRepresentation(features);

		double[][] DTW = new double[n][m];

		for (int i = 0; i < n; i++)
			DTW[i][0] = Double.POSITIVE_INFINITY;
		for (int i = 0; i < m; i++)
			DTW[0][i] = Double.POSITIVE_INFINITY;
		DTW[0][0] = 0;

		for (int i = 1; i < n; i++) {
			for (int j = 1; j < m; j++) {
				double cost = 0;
				for(int f = 0; f < features.size(); f++){
					cost += Math.pow(sArray[i-1][f] - tArray[j-1][f], 2);
				}
				cost = Math.sqrt(cost);
				DTW[i][j] = cost + Math.min(DTW[i - 1][j], Math.min( // insertion
						DTW[i][j - 1], // deletion
						DTW[i - 1][j - 1])); // match
			}
		}

		return DTW[n - 1][m - 1];
	}

}
