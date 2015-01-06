package ch.eonum.pipeline.distance;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import ch.eonum.pipeline.core.Instance;

/**
 * Cosine Distance. Inverse of cosine similarity: {@link http://en.wikipedia.org/wiki/Cosine_similarity}
 * @author tim
 *
 * @param <E>
 */
public class CosineDistance<E extends Instance> extends Distance<E> {

	public CosineDistance() {
		super(null);
	}

	@Override
	public double distance(E inst1, E inst2) {
		double sumTop = 0;
        double sumOne = 0;
        double sumTwo = 0;
        /** merge dimensions. */
		List<String> union = new ArrayList<String>(inst2.features());
		union.addAll(inst1.features());
		for(String feature : new HashSet<String>(union)) {
            sumTop += inst1.get(feature) * inst2.get(feature);
            sumOne += inst1.get(feature) * inst1.get(feature);
            sumTwo += inst2.get(feature) * inst2.get(feature);
        }
        double cosSimilarity = sumTop / (Math.sqrt(sumOne) * Math.sqrt(sumTwo));
        if (cosSimilarity < 0)
            cosSimilarity = 0; /** This should not happen, but does due to rounding errors. */
        
        return 1 - cosSimilarity;
	}
	
	@Override
	public double length(E inst){
		return 1.0;
	}

}
