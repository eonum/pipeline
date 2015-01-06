package ch.eonum.pipeline.distance;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;

/**
 * Minkowski distance. Generalization of the Euclidian distance.
 * The parameter k, which is always 2 in the Euclidian distance, 
 * is a free parameter in the Minkowski distance.
 * 
 * d = ((x1 - y1)^k + ... + (xn - yn)^k)^(1/k)
 *  
 * @author tim
 *
 */
public class MinkowskiDistance<E extends Instance> extends Distance<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("minkowski", "minkowski constant for the minkowski metric, 2.0 => euclidian metric" +
				":  (default 2.0)");
	}

	@SuppressWarnings("unchecked")
	public MinkowskiDistance(double m) {
		super((E)new SparseInstance("", "", new HashMap<String, Double>()));
		this.setSupportedParameters(MinkowskiDistance.PARAMETERS);
		this.setMinkowskiParameter(m);
	}

	public void setMinkowskiParameter(double m) {
		this.putParameter("minkowski", m);
	}

	@Override
	public double distance(Instance inst1, Instance inst2, double lengthInstance2) {
		double minkowski = getDoubleParameter("minkowski");
		double distance = lengthInstance2;
		for(String feature : inst1.features())
			distance += Math.pow(Math.abs(
					inst2.get(feature) - inst1.get(feature)), minkowski)
					- Math.abs(Math.pow(inst2.get(feature), minkowski));
		
		if(distance < 0.000000000001) return 0.0;// avoid NaN
		return Math.abs(Math.pow(distance, 1/minkowski));
	}

	@Override
	public double distance(Instance inst1, Instance inst2) {
		double minkowski = getDoubleParameter("minkowski");
		double distance = 0.0;
		List<String> union = new ArrayList<String>(inst2.features());
		union.addAll(inst1.features());
		for(String feature : new HashSet<String>(union))
			distance += Math.pow(Math.abs(
					inst2.get(feature) - inst1.get(feature)), minkowski);
		
		if(distance < 0.000000000001) return 0.0;// avoid NaN
		return Math.abs(Math.pow(distance, 1/minkowski));
	}

}
