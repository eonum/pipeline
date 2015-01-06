package ch.eonum.pipeline.util.json;

import java.util.Map;

/**
 * JSON - Serialization. Interface for all objects which can be written to and
 * read from json files/streams
 * 
 * @author tim
 * 
 */
public interface JSONable {
	/**
	 * Serialization.
	 * Get the object in a tree representation using only simple objects. The
	 * tree is used for Jackson's Simple Data Binding.
	 * 
	 * @return a tree representation of this object.
	 */
	public Map<String, Object> asTree();
	
	/**
	 * Deserialization.
	 * Read a tree into this object.
	 * @param tree
	 * @throws IllegalAccessException
	 * @throws InstantiationException 
	 * @throws ClassNotFoundException 
	 */
	public void readFromTree(Map<String, Object> tree)
			throws InstantiationException, IllegalAccessException,
			ClassNotFoundException;
}
