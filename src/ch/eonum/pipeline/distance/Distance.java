package ch.eonum.pipeline.distance;


import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Parameters;

/**
 * measures a distance or a similarity between two classes.
 * @author tim
 *
 */
public abstract class Distance<E extends Instance> extends Parameters {

	protected E zeroInstance;
	
	/**
	 * @param zi Zero instance.
	 */
	public Distance(E zi){
		this.zeroInstance = zi;
	}

	/**
	 * get the distance between two instances.
	 * @param inst1
	 * @param inst2
	 * @return
	 */
	public abstract double distance(E inst1, E inst2);
	
	/**
	 * get the distance between two instances.
	 * This gives exactly the same result as the standard distance.
	 * Use this method if you have one instance with many features (e.g. cluster center)
	 * and one with only few. You have to provide the precomputed length of instance2.
	 * This is much faster in some cases and for some distances (but not all) than the standard distance.
	 * 
	 * @param inst1
	 * @param inst2
	 * @param lengthOfInstance2
	 * @return
	 */
	public double distance(E inst1, E inst2, double lengthOfInstance2){
		return distance(inst1, inst2);
	}
	
	/**
	 * calculate the length of an instance.
	 * the length is the same as the distance to a 0-instance.
	 * @param instance
	 * @return length
	 */
	public double length(E inst) {
		return distance(this.zeroInstance, inst);
	}

}
