package ch.eonum.pipeline.distance;

import java.util.HashMap;
import java.util.Map;

import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;

/**
 * Implementation of a polynomial kernel.
 * A polynomial kernel is not a metric!! But can be used as a similarity measure.
 * 
 * @author tim
 *
 */
public class PolynomialKernel<E extends Instance> extends Distance<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("exponent", "exponent for the polynomial kernel" +
				":  (default 1.0)");
	}
	
    @SuppressWarnings("unchecked")
	public PolynomialKernel(){
		super((E)new SparseInstance("", "", new HashMap<String, Double>()));
    	this.setSupportedParameters(PolynomialKernel.PARAMETERS);
    	this.putParameter("exponent", 1.0);
    }
    
    @Override
    public double distance(Instance i, Instance j) {
    	double exponent = this.getDoubleParameter("exponent");
        double result;
        result = dotProd(i, j);
        if (exponent != 1.0) {
            result = Math.pow(result, exponent);
        }
        return 1 - result;
    }

    /**
     * Calculates a dot product between two instances
     */
    protected final double dotProd(Instance inst1, Instance inst2) {
        double result = 0;
        Instance lessDimensions = inst1.features().size() > inst2.features().size() ? inst2 : inst1;
        for (String feature : lessDimensions.features()) {
            result += inst1.get(feature) * inst2.get(feature);
        }
        return result;
    }

	@Override
	public double length(Instance inst){
		return 1.0;
	}

}
