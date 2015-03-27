package ch.eonum.pipeline.core;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import ch.eonum.pipeline.util.Log;
import ch.eonum.pipeline.util.json.JSON;

/**
 * Parameter object providing meta parameters. Every object which has meta
 * parameters, can inherit Parameters. A meta parameter is a parameter which
 * is/can be validated during validation.
 * 
 * @author tp
 * 
 */
public class Parameters {

	private Map<String, Double> doubleParameters;
	private Map<String, Integer> intParameters;
	private Map<String, String> stringParameters;
	private Map<String, Boolean> booleanParameters;
	
	/**
	 * Constructor. Initialize empty parameter sets.
	 */
	public Parameters() {
		this.doubleParameters = new HashMap<String, Double>();
		this.intParameters = new HashMap<String, Integer>();
		this.stringParameters = new HashMap<String, String>();
		this.booleanParameters = new HashMap<String, Boolean>();
		this.setSupportedParameters(new HashMap<String, String>());
	}
	
	/**
	 * a list with all supported parameters and their description (should also
	 * contain the default value)
	 */
	private Map<String, String> supportedParameters;
	
	/**
	 * Check if the used parameter is supported. If the parameter is not
	 * supported the application exits.
	 * 
	 * @param param
	 */
	private void checkParam(String param) {
		if(!supportedParameters.containsKey(param))
			Log.error("The parameter " + param + " is not supported");
	}

	/**
	 * Set a double parameter.
	 * 
	 * @param param
	 * @param value
	 */
	public void putParameter(String param, double value) {
		checkParam(param);
		this.doubleParameters.put(param, value);
	}
	
	/**
	 * Set an integer parameter.
	 * 
	 * @param param
	 * @param value
	 */
	public void putParameter(String param, int value) {
		checkParam(param);
		this.intParameters.put(param, value);
	}
	
	/**
	 * Set a string parameter.
	 * 
	 * @param param
	 * @param value
	 */
	public void putParameter(String param, String value) {
		checkParam(param);
		this.stringParameters.put(param, value);
	}
	
	/**
	 * Set a boolean parameter.
	 * 
	 * @param param
	 * @param value
	 */
	public void putParameter(String param, boolean value) {
		checkParam(param);
		this.booleanParameters.put(param, value);
	}

	/**
	 * Get the value of double parameter param.
	 * 
	 * @param param
	 * @return
	 */
	public synchronized double getDoubleParameter(String param) {
		return this.doubleParameters.get(param);
	}
	
	/**
	 * Get the value of integer parameter param.
	 * @param param
	 * @return
	 */
	public synchronized int getIntParameter(String param) {
		return this.intParameters.get(param);
	}
	
	/**
	 * Get the value of string parameter param.
	 * 
	 * @param param
	 * @return
	 */
	public synchronized String getStringParameter(String param) {
		return this.stringParameters.get(param);
	}
	
	/**
	 * Get the value of boolean parameter param. 
	 * 
	 * @param param
	 * @return
	 */
	public synchronized boolean getBooleanParameter(String param) {
		return this.booleanParameters.get(param);
	}
	
	/**
	 * print all parameters into a file.
	 * 
	 * @param file
	 * @throws IOException 
	 */
	public void printParameters(File file) throws IOException {
		JSON.writeJSON(file, this.asMap());
	}

	/**
	 * Get all parameters as a nested map containing one sub map for each
	 * parameter type. Used for JSON-serialization.
	 * 
	 * @return
	 */
	public Map<String, Object> asMap() {
		Map<String, Object> dbo = new HashMap<String, Object>();
		dbo.put("stringParams", this.stringParameters);
		dbo.put("doubleParams", this.doubleParameters);
		dbo.put("intParams", this.intParameters);
		dbo.put("booleanParams", this.booleanParameters);
		return dbo;
	}

	/**
	 * Print parameters to file.
	 * @param fileName
	 * @throws IOException
	 */
	public void printParameters(String fileName) throws IOException{
		this.printParameters(new File(fileName));
	}
	
	/**
	 * Load parameters from file.
	 * @param file
	 * @throws IOException 
	 */
	public void load(File file) throws IOException {
		Map<String, Object> data = JSON.readJSON(file);
		this.load(data);
	}
	
	/**
	 * Load parameters from a map as produced by {@link #asMap()}. 
	 * @param data
	 */
	@SuppressWarnings("unchecked")
	protected void load(Map<String, Object> data) {
		this.booleanParameters = (Map<String, Boolean>) data.get("booleanParams");
		this.stringParameters = (Map<String, String>) data.get("stringParams");
		this.doubleParameters = (Map<String, Double>) data.get("doubleParams");
		this.intParameters = (Map<String, Integer>) data.get("intParams");
	}

	/**
	 * load parameters from file as produced by
	 * {@link #printParameters(File file)} (JSON).
	 * 
	 * @param fileName
	 * @throws IOException
	 */
	public void load(String fileName) throws IOException {
		this.load(new File(fileName));
	}

	/**
	 * Set a map of supported parameters (keys) and their description (values).
	 * 
	 * @param supportedParameters
	 */
	public void setSupportedParameters(Map<String, String> supportedParameters) {
		this.supportedParameters = supportedParameters;
	}

	/**
	 * Get a map of supported parameters (keys) and their description (values).
	 * 
	 * @return
	 */
	public Map<String, String> getSupportedParameters() {
		return supportedParameters;
	}
}
