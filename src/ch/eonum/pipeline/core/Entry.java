package ch.eonum.pipeline.core;

/**
 * Entry in a sparse array. Implemented as an immutable object.
 * @author tim
 *
 */
public final class Entry {
	public Entry(int index, double value){
		this.index = index;
		this.value = value;
	}
	/** vector index. */
	public final int index;
	/** value at this index. */
	public final double value;
	
	@Override
	public String toString(){
		return this.index + " => " + this.value;
	}
}
