/*
 * To change this template, choose Tools | Templates
 * and openTableFileImageItem the template in the editor.
 */
package xmipp.viewer.windows;

import xmipp.utils.DEBUG;
import xmipp.utils.InfiniteProgressPanel;
import xmipp.utils.Param;
import xmipp.viewer.JFrameGallery;
import xmipp.viewer.imageitems.tableitems.AbstractGalleryImageItem;
import xmipp.viewer.gallery.models.AbstractXmippTableModel;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.ImageWindow;
import ij.gui.Toolbar;
import ij.io.FileInfo;
import ij.process.StackConverter;
import ij3d.Content;
import ij3d.Image3DUniverse;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Toolkit;
import java.io.File;
import java.util.ArrayList;
import javax.swing.JFrame;
import javax.swing.JRootPane;
import javax.vecmath.Color3f;
import xmipp.viewer.metadata.JFrameMetaData;
import xmipp.viewer.micrographs.JFrameMicrographs;
import xmipp.viewer.micrographs.ctf.CTFRecalculateImageWindow;
import xmipp.viewer.micrographs.CTFProfileWindow;
import xmipp.viewer.micrographs.FSCWindow;
import xmipp.viewer.micrographs.ctf.tasks.TasksEngine;
import xmipp.viewer.rotspectra.JFrameRotSpectra;
import xmipp.jni.Filename;
import xmipp.jni.ImageGeneric;
import xmipp.jni.MDLabel;
import xmipp.jni.MetaData;
import xmipp.ij.XmippImageConverter;

/**
 *
 * @author Juanjo Vega
 */
public class ImagesWindowFactory {

    private final static int UNIVERSE_W = 400, UNIVERSE_H = 400;
//    private final static String TEMPDIR_PATH = System.getProperty("java.io.tmpdir");

    public static void blockGUI(JRootPane panel, String status) {
        final InfiniteProgressPanel progressPanel = new InfiniteProgressPanel(status);
        panel.setGlassPane(progressPanel);
        progressPanel.start();
    }

    public static void releaseGUI(JRootPane panel) {
        InfiniteProgressPanel progressPanel = (InfiniteProgressPanel) panel.getGlassPane();
        progressPanel.stop();
        progressPanel.setVisible(false);
    }

    public static void openFilesAsDefault(String filenames[], Param parameters) throws Exception {
        for (int i = 0; i < filenames.length; i++) {
            openFileAsDefault(filenames[i], parameters);
        }
    }

    public static void openFileAsDefault(String filenames) throws Exception {
        openFileAsDefault(filenames, new Param());
    }

    public static void openFileAsDefault(String filename, Param parameters) throws Exception {
        if (Filename.isMetadata(filename)) {
            openMetadata(filename, parameters, Param.OPENING_MODE_GALLERY);
        } else {
            try {
                ImageGeneric img = new ImageGeneric(filename);

                if (img.isSingleImage()) {
                    openFileAsImage(filename, parameters);
                } else if (img.isStack()) {
                    openMetadata(filename, parameters,Param.OPENING_MODE_GALLERY);
                } else {
                    openFileAsImage(filename, parameters);
                }
            } catch (Exception e) {
                IJ.open(filename);
                //IJ.error(e.getMessage());
            }
        }
    }

    public static void openFilesAsImages(String filenames[], Param parameters) {
        for (int i = 0; i < filenames.length; i++) {
            String filename = Filename.getFilename(filenames[i]);
            long nimage = Filename.getNimage(filenames[i]);

            DEBUG.printMessage(" *** Opening: " + filename + " / nimage: " + nimage);

            openFileAsImage(filenames[i], parameters);
        }
    }

    public static void openFileAsImage(String path) {
        openFileAsImage(path, new Param());
    }

    public static void openFileAsImage(String path, Param parameters) {
        try {
            ImagePlus imp;

            imp=openFileAsImagePlus(path, parameters);

            // Normalize image stack.
            //XmippImageConverter.normalizeImagePlus(imp);

            openXmippImageWindow(imp, parameters.poll);
        } catch (Exception ex) {
            IJ.error(ex.getMessage() + ": " + path);
            DEBUG.printException(ex);
        }
    }
    
    public static ImagePlus openFileAsImagePlus(String path, Param parameters) throws Exception{
        ImagePlus imp;
        if (Filename.isMetadata(path)) {
            MetaData md = new MetaData(path);

            imp = XmippImageConverter.convertToImageJ(md);
        } else {
            imp = XmippImageConverter.loadImage(path, parameters.zoom);
        }
        return imp;
    }

    private static ImageWindow openXmippImageWindow(ImagePlus imp, boolean poll) {
        ImageWindow iw = null;

        if (imp != null) {
            if (imp.getStackSize() > 1) {
                iw = new StackWindowOperations(imp, poll);
            } else {
                iw = new ImageWindowOperations(imp, poll);
            }
        }

        return iw;
    }

//    public static void openFilesAsMetadata(String filenames[]) {
//        openFilesAsMetadata(filenames, new Param());
//    }
//
//    public static void openFilesAsMetadata(String filenames[], Param parameters) {
//        for (int i = 0; i < filenames.length; i++) {
//            openFileAsMetadata(filenames[i], parameters);
//        }
//    }
//
//    public static void openFileAsMetadata(String filename) {
//        openFileAsMetadata(filename, new Param());
//    }
//
//    public static void openFileAsMetadata(String filename, MetaData md, Param parameters) {
//    	parameters.mode = Param.OPENING_MODE_METADATA;
//    	openFileAsGallery(filename, parameters);
//    }
//
//    public static void openMetadataAsGallery(String filename, MetaData md) {
//        openFileAsGallery(filename, md, new Param(), Param.OPENING_MODE_GALLERY);
//    }

    /** Before calling this method be sure you have constructed
     * the proper metadata with files to be shown, mode passed
     * will be override in parameters
     */
    public static JFrameGallery openMetadata(String filename, MetaData md, 
    		Param parameters, String mode) {
    	if (parameters.debug)
    		DEBUG.enableDebug(true);
    	if (parameters.mode.equalsIgnoreCase(Param.OPENING_MODE_DEFAULT))
    		parameters.mode = mode;
        return new JFrameGallery(filename, md, parameters);
    }
    
    public static JFrameGallery openMetadata(String filename, Param parameters,
    		String mode) throws Exception{
    	return openMetadata(filename, new MetaData(filename), parameters, mode);
    }

    public static JFrameGallery openFilesAsGallery(String filenames[], boolean useSameTable) throws Exception {
        return openFilesAsGallery(filenames, useSameTable, new Param());
    }

    public static JFrameGallery openFilesAsGallery(String filenames[],
            boolean useSameTable, Param parameters) throws Exception {
        JFrameGallery gallery = null;

        if (useSameTable) {
        	MetaData md = new MetaData();
        	for (int i = 0; i < filenames.length; ++i)
        		md.setValueString(MDLabel.MDL_IMAGE, filenames[i], md.addObject());
        	openMetadata(null, md, parameters, null);
        } else {
            for (int i = 0; i < filenames.length; i++) {
                gallery = openMetadata(filenames[i], parameters, Param.OPENING_MODE_GALLERY);
            }
        }

        return gallery;
    }

    // Used by micrographs table, to load items marked as selected/unselected.
    public static void openGallery(String filenames[], boolean enabled[]) {
//        JFrameGallery gallery = new JFrameGallery(filenames, enabled);
//        gallery.setAutoAdjustColumns(true);
//        gallery.setVisible(true);
    }

    public static void captureFrame(ImagePlus ip) {
        openXmippImageWindow(ip, false);
    }

    public static void openGalleryAs3D(AbstractXmippTableModel tableModel) {
        try {
//            ArrayList<AbstractGalleryImageItem> items = tableModel.getAllItems();
//            ImagePlus ip = ImagesWindowFactory.convertToImageJ(items);
//            ip.setTitle(tableModel.getFilename());

           // openImagePlusAs3D(ip);
        } catch (Exception ex) {
            IJ.error(ex.getMessage());
            DEBUG.printException(ex);
        }
    }

    public static void openGalleryAsImagePlus(AbstractXmippTableModel tableModel) {
        try {
            String path = tableModel.getFilename();

            // If file exists, uses it...
            File file = new File(Filename.getFilename(path));
            if (file.exists()) {
//                System.err.println(" +++ EXISTS");
                openFileAsImage(path, new Param());
            } else {
//                System.err.println(" !!! EXISTS");
                // ...otherwise, stores it in a temporary file.
                File tempFile = File.createTempFile("tableToStack_", ".stk");
                tempFile.deleteOnExit();

//                ArrayList<AbstractGalleryImageItem> items = tableModel.getAllItems();
//                ImagePlus imp = ImagesWindowFactory.convertToImageJ(items);
//                IJ.run(imp, "Xmipp writer", "save=" + tempFile.getAbsolutePath());

//                System.err.println(" >>> TMP Saved at: " + file.getAbsolutePath());
//
//                imp.setTitle(tempFile.getName());
//
//                captureFrame(imp);
            }
        } catch (Exception ex) {
            IJ.error(ex.getMessage());
            DEBUG.printException(ex);
        }
    }

    public static void openImagePlusAsGallery(ImagePlus imp) {
        try {
            FileInfo fi = imp.getOriginalFileInfo();

            // If path exists, uses it...
            File file = null;
            if (fi != null && !fi.fileName.trim().isEmpty() && !fi.directory.trim().isEmpty()) {
                file = new File(fi.directory + File.separator + fi.fileName);
            }

            if (file == null || !file.exists()) {   // ...otherwise, stores it in a temporary file.
                file = File.createTempFile("stackToTable_", ".stk");
                file.deleteOnExit();
                
                XmippImageConverter.saveImage(imp, file.getAbsolutePath());
            }

            openMetadata(file.getAbsolutePath(), new Param(), Param.OPENING_MODE_GALLERY);
        } catch (Exception ex) {
            IJ.error(ex.getMessage());
            DEBUG.printException(ex);
        }
    }

    public static void openImagePlusAs3D(ImagePlus ip) {
        try {
            // Checks if java3D is available or not.
            Class.forName("javax.media.j3d.J3DBuffer");

            new StackConverter(ip).convertToRGB();

            Image3DUniverse universe = new Image3DUniverse(UNIVERSE_W, UNIVERSE_H);

            // Adds the sphere image plus to universe.
            Content c = universe.addSurfacePlot(ip, new Color3f(1f, 165f / 255, 82f / 255), "1",
                    50, new boolean[]{true, true, true}, 1);
            c.displayAs(Content.SURFACE);
            c.setColor(new Color3f(1f, 165f / 255, 82f / 255));

            universe.show();    // Shows...
        } catch (final ClassNotFoundException e) {
            IJ.error("Java 3D not found. Please, check your installation.");
        }
    }

    public static ImageWindow openCTFImage(ImagePlus ip, String CTFfilename,
            String PSDfilename, TasksEngine tasksEngine,
            String MicrographFilename, int row) {
        IJ.setTool(Toolbar.FREEROI);

        return new CTFRecalculateImageWindow(ip, CTFfilename, PSDfilename,
                tasksEngine, row);
    }

    public static void openMicrograph(String filename) {
        File f = new File(filename);

        if (f.exists()) {
            JFrameMicrographs frame = new JFrameMicrographs(filename);
            frame.setVisible(true);
        } else {
            IJ.error("File is missing", filename + " not found.");
        }
    }

    public static void openRotSpectraWindow(String filenameVectors,
            String filenameClasses, String filenameData) {
        JFrameRotSpectra frame = new JFrameRotSpectra(filenameVectors, filenameClasses, filenameData);
        ImagesWindowFactory.setConvenientSize(frame);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    public static void openFileAsText(String filename, Component parent) {
        JFrameTextFile frameCTF = new JFrameTextFile(filename);
        frameCTF.setLocationRelativeTo(parent);
        frameCTF.setVisible(true);
    }

    public static void openCTFWindow(ImagePlus imp, String CTFFilename, String PSDFilename) {
        CTFProfileWindow ctfView = new CTFProfileWindow(imp, CTFFilename, PSDFilename);
        ctfView.setVisible(true);
    }

    public static void openFSCWindow(String filename) {
        FSCWindow frame = new FSCWindow(filename);
        frame.setVisible(true);
    }

    public static String getSortTitle(String title, int width, FontMetrics fontMetrics) {
        String sort = title;
        int strlenght = fontMetrics.stringWidth(sort);
        int index = 0;

        while (strlenght > width) {
            index++;
            sort = "..." + title.substring(index);
            strlenght = fontMetrics.stringWidth(sort);
        }

        return sort;
    }

    public static void setConvenientSize(JFrame frame) {
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        int w = screenSize.width * 2 / 3;
        int h = screenSize.height * 2 / 3;

        frame.setSize(w, h);
        frame.setLocationRelativeTo(null);
    }

    public static ImagePlus convertToImageJ(ArrayList<AbstractGalleryImageItem> items) {
        ImageStack is = null;

        for (int i = 0; i < items.size(); i++) {
            AbstractGalleryImageItem item = items.get(i);

            if (item.isEnabled() && item.exists()) {
                ImagePlus ipslice = item.getImagePlus();

                if (is == null) {
                    is = new ImageStack(ipslice.getWidth(), ipslice.getHeight());
                }

                is.addSlice(ipslice.getTitle(), ipslice.getProcessor());
            }
        }

        return new ImagePlus("", is);
    }
}
