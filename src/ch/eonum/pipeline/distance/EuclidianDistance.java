package ch.eonum.pipeline.distance;

import ch.eonum.pipeline.core.Instance;

/**
 * Implementation of the Euclidian distance, which is just a special case
 * of the Minkowski distance with k = 2
 * 
 * d = ((x1 - y1)^2 + ... + (xn - yn)^2)^(1/2)
 * 
 * @author tim
 *
 */
public class EuclidianDistance<E extends Instance> extends MinkowskiDistance<E> {

	public EuclidianDistance() {
		super(2);
	}

}
