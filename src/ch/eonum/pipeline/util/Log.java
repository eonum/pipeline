package ch.eonum.pipeline.util;


/**
 * Logger class.
 * @author tim
 *
 */
public class Log {

	public static void puts(Object object){
		System.out.println(object);
	}
	
	public static void warn(Object object){
		System.err.println("Warning: " + object);
	}
	
	public static void error(Object object){
		System.err.println("Error: " + object);
		System.exit(-1);
	}

	public static void printf(String format, Object ... args) {
		System.out.printf(format, args);
	}
}
