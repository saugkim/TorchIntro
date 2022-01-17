package org.tuni.mobilenet;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.content.ContentResolver;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

    final static String MODEL_NAME = "mobilenet_scripted.pt";
    ImageView imageView;
    TextView textView;
    Module module = null;
    Bitmap mBitmap = null;
    String mName = "A";
    String filename;
    Button galleryButton, runButton;

    ActivityResultLauncher<String> mGetContent;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private void getImageFromDeviceGallery() {
        mGetContent.launch("image/*");
    }

    private Bitmap getBitmapFromUri(Uri imageUri) {
        Bitmap bitmap = null;
        ContentResolver contentResolver = getContentResolver();
        try {
            if(Build.VERSION.SDK_INT < 28) {
                bitmap = MediaStore.Images.Media.getBitmap(contentResolver, imageUri);
            } else {
                ImageDecoder.Source source = ImageDecoder.createSource(contentResolver, imageUri);
                bitmap = ImageDecoder.decodeBitmap(source).copy(Bitmap.Config.RGBA_F16, true);
            }
            return centerCropAndResizeBitmapImage(bitmap);
            //return bitmap;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);
        filename = getString(R.string.image_name);
        galleryButton = findViewById(R.id.galleryButton);
        runButton = findViewById(R.id.runButton);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(filename));
        } catch (IOException e) {
            Log.e("IMAGE NOT FOUND", "Error open image!", e);
            finish();
        }
        imageView.setImageBitmap(mBitmap);

        try {
            module = Module.load(assetFilePath(this, MODEL_NAME));
        } catch (IOException e) {
            Log.e("MODEL NOT FOUND", "Error loading model!", e);
            finish();
        }

        mGetContent = registerForActivityResult(new ActivityResultContracts.GetContent(),
                uri -> {
                    // imageView.setImageURI(uri);
                    //Log.d("gallery uri", uri.toString());
                    mBitmap = getBitmapFromUri(uri);
                    imageView.setImageBitmap(mBitmap);
                });

        galleryButton.setOnClickListener(v -> getImageFromDeviceGallery());

        runButton.setOnClickListener(v-> {
            mName = runModel(mBitmap);
            textView.setText(mName);
        });
    }

    private String runModel(Bitmap mBitmap) {
        // preparing input tensor
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

        // running the model
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // getting tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray();

        // searching for the index with maximum score
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }
        return ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];
    }

    Bitmap centerCropAndResizeBitmapImage(Bitmap srcBmp) {
        Bitmap dstBmp;
        int ret_size = 0;
        if (srcBmp.getWidth() >= srcBmp.getHeight()){
            ret_size = srcBmp.getHeight();
            dstBmp = Bitmap.createBitmap(
                    srcBmp,
                    srcBmp.getWidth()/2 - srcBmp.getHeight()/2,
                    0,
                    srcBmp.getHeight(),
                    srcBmp.getHeight()
            );
        } else {
            ret_size = srcBmp.getWidth();
            dstBmp = Bitmap.createBitmap(
                    srcBmp,
                    0,
                    srcBmp.getHeight()/2 - srcBmp.getWidth()/2,
                    ret_size,
                    ret_size
            );
        }

        if (ret_size < 244){
            return Bitmap.createScaledBitmap(dstBmp, 244, 244, false);
        } else {
            return Bitmap.createScaledBitmap(dstBmp, 400, 400, false);
        }
    }
}