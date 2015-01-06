package ch.eonum.pipeline.util.json;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.codehaus.jackson.JsonGenerationException;
import org.codehaus.jackson.JsonParseException;
import org.codehaus.jackson.map.JsonMappingException;
import org.codehaus.jackson.map.ObjectMapper;
import org.codehaus.jackson.type.TypeReference;


/**
 * JSON - Serialization. Wrapper class for all Jackson JSON functionalities.
 * 
 * @author tim peter
 * 
 */
public class JSON {
	/**
	 * Write a JSONable object to a File.
	 * JSON is pretty printed.
	 * 
	 * @param file
	 * @param object JSONable object
	 * @throws JsonGenerationException
	 * @throws JsonMappingException
	 * @throws IOException
	 */
	public static void writeJSON(File file, JSONable object)
			throws JsonGenerationException, JsonMappingException, IOException {
		JSON.writeJSON(file, object.asTree());
	}
	
	/**
	 * Write a map to a File as JSON.
	 * JSON is pretty printed.
	 * 
	 * @param file
	 * @param object object tree representation
	 * @throws JsonGenerationException
	 * @throws JsonMappingException
	 * @throws IOException
	 */
	public static void writeJSON(File file, Map<String, Object> object)
			throws JsonGenerationException, JsonMappingException, IOException {
		ObjectMapper mapper = new ObjectMapper();
		mapper.writerWithDefaultPrettyPrinter().writeValue(file, object);
	}

	/**
	 * Read a JSON file.
	 * 
	 * @param file
	 * @return
	 * @throws JsonParseException
	 * @throws JsonMappingException
	 * @throws IOException
	 */
	public static Map<String, Object> readJSON(File file) throws IOException {
		ObjectMapper mapper = new ObjectMapper();
		return mapper.readValue(file, new TypeReference<Map<String, Object>>() { });
	}

	/**
	 * Read a JSON file.
	 * 
	 * @param file fileName
	 * @return
	 * @throws JsonParseException
	 * @throws JsonMappingException
	 * @throws IOException
	 */
	public static Map<String, Object> readJSON(String fileName)
			throws JsonParseException, JsonMappingException, IOException {
		return JSON.readJSON(new File(fileName));
	}
}
