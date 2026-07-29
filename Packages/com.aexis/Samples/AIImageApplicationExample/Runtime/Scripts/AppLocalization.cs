using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

public enum AppLanguage
{
    English,
    SimplifiedChinese
}

public static class AppLocalization
{
    private const string LanguagePrefKey = "AIImage.Language";

    // Keep source ASCII-only so the lookup remains stable across package importers.
    private static readonly Dictionary<string, string> ChineseToEnglish = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["\u5386\u53f2\u8bb0\u5f55"] = "History",
        ["\u4e3b\u7f16\u8f91\u5386\u53f2"] = "Edit history",
        ["\u8bbe\u8ba1\u5386\u53f2"] = "Design history",
        ["\u539f\u56fe: "] = "Original: ",
        ["\u539f\u59cb\u56fe\u50cf\u4e0d\u80fd\u5220\u9664"] = "The original image cannot be deleted.",
        ["\u64a4\u9500"] = "Undo",
        ["\u5220\u9664\u5f53\u524d"] = "Delete current",
        ["\u8bf7\u9009\u62e9\u7ed3\u679c"] = "Choose a result",
        ["\u53d6\u6d88"] = "Cancel",
        ["\u7ed3\u679c "] = "Result ",
        ["\u5904\u7406\u4e2d"] = "Processing",
        ["\u56fe\u5e93"] = "Library",
        ["\u540d\u79f0"] = "Name",
        ["\u4eba\u8138"] = "Faces",
        ["\u5730\u70b9"] = "Location",
        ["\u539f\u56fe"] = "Original",
        ["\u4fee\u56fe"] = "Edited",
        ["\u672a\u77e5"] = "Unknown",
        ["\u6536\u85cf"] = "Favorites",
        ["\u6309\u540d\u79f0\u6392\u5e8f"] = "Sort by name",
        ["\u6309\u4eba\u8138\u6392\u5e8f"] = "Sort by faces",
        ["\u6309\u5730\u70b9\u6392\u5e8f"] = "Sort by location",
        ["\u663e\u793a\u539f\u56fe"] = "Show originals",
        ["\u663e\u793a\u4fee\u56fe"] = "Show edited",
        ["\u663e\u793a\u672a\u77e5\u7c7b\u578b"] = "Show unknown",
        ["\u4ec5\u663e\u793a\u6536\u85cf"] = "Show favorites only",
        ["\u8bf7\u9009\u62e9\u76ee\u5f55"] = "Select a folder",
        ["\u7f29\u7565\u56fe\u4fe1\u606f"] = "Thumbnail details",
        ["\u5355\u51fb\u7f29\u7565\u56fe\u67e5\u770b\u4fe1\u606f\uff0c\u53cc\u51fb\u76f4\u63a5\u8fdb\u5165\u4e3b\u7f16\u8f91\u9875\u3002"] = "Click a thumbnail for details. Double-click to open it in the editor.",
        ["\u6620\u5c04\u539f\u56fe:"] = "Mapped original:",
        ["\u5c55\u5f00\u56fe\u7247\u8be6\u60c5"] = "Expand image details",
        ["\u6536\u8d77\u56fe\u7247\u8be6\u60c5"] = "Collapse image details",
        ["\u76d8\u7b26"] = "Drive",
        ["\u5b58\u50a8"] = "Storage",
        ["\u4f4d\u7f6e"] = "Location",
        ["\u672c\u5730"] = "Local",
        ["\u5916\u63a5"] = "External",
        ["\u6302\u8f7d"] = "Mounted",
        ["\u5e94\u7528\u6587\u4ef6"] = "App files",
        ["\u56fe\u7247"] = "Pictures",
        ["\u5f85\u63d0\u53d6"] = "Pending",
        ["\u5f85\u63a5\u5165"] = "Pending",
        ["\u65e0"] = "None",
        ["\u8be5\u76ee\u5f55\u4e0d\u5b58\u5728\u3002"] = "This folder does not exist.",
        ["\u6b63\u5728\u626b\u63cf\u76ee\u5f55..."] = "Scanning folder...",
        ["\u76ee\u5f55\u626b\u63cf\u5931\u8d25\u3002"] = "Folder scan failed.",
        ["\u5f53\u524d\u7b5b\u9009\u4e0b\u6ca1\u6709\u56fe\u7247\u3002"] = "No images match the current filters.",
        ["\u62cd\u6444\u65f6\u95f4: "] = "Captured: ",
        ["\u6587\u4ef6\u5927\u5c0f: "] = "File size: ",
        ["\u5730\u70b9: "] = "Location: ",
        ["\u76f8\u673a: "] = "Camera: ",
        ["\u5149\u5708: "] = "Aperture: ",
        ["\u4eba\u8138: "] = "Faces: ",
        ["\u6ca1\u6709\u53ef\u5b9a\u4f4d\u7684\u539f\u56fe"] = "No original image can be located.",
        ["\u65e0\u6cd5\u9884\u89c8"] = "Preview unavailable",
        ["\u52a0\u8f7d\u4e2d..."] = "Loading...",
        ["\u7b49\u5f85\u52a0\u8f7d"] = "Waiting to load",
        ["\u4fee\u8fc7\u56fe"] = "Edited image",
        ["\u6620\u5c04\u539f\u56fe"] = "Mapped original",
        ["\u539f\u56fe\u6587\u4ef6\u4e0d\u5b58\u5728"] = "Original image does not exist.",
        ["\u65e0\u6cd5\u6253\u5f00\u539f\u56fe\u4f4d\u7f6e"] = "Unable to open the original image location.",
        ["\u65e0\u6cd5\u8bbf\u95ee\u5b58\u50a8\uff0c\u8bf7\u6388\u4e88\u56fe\u7247\u6216\u6587\u4ef6\u8bfb\u53d6\u6743\u9650"] = "Storage access requires image or file read permission.",
        ["\u6b63\u5728\u68c0\u67e5\u5b58\u50a8\u8bbf\u95ee..."] = "Checking storage access...",
        ["\u5f53\u524d\u5b58\u50a8\u4f4d\u7f6e\u65e0\u6cd5\u8bbf\u95ee"] = "The current storage location is unavailable.",
        ["\u76ee\u5f55\u53ef\u89c1\uff0c\u4f46\u5f53\u524d\u7cfb\u7edf\u672a\u8fd4\u56de\u6587\u4ef6\u5217\u8868"] = "The folder is visible, but the system did not return a file list.",
        ["\u68c0\u67e5\u5b58\u50a8\u8bbf\u95ee\u5931\u8d25"] = "Storage access check failed.",
        ["\u8be5\u76ee\u5f55\u53ef\u5c55\u5f00\uff0c\u4f46\u5f53\u524d\u7cfb\u7edf\u672a\u8fd4\u56de\u6587\u4ef6\u5217\u8868"] = "This folder can be expanded, but the system did not return a file list.",
        ["\u5f53\u524d\u76ee\u5f55\u6ca1\u6709\u8bfb\u53d6\u6743\u9650"] = "The current folder cannot be read.",
        ["\u8bfb\u53d6\u76ee\u5f55\u5185\u5bb9\u5931\u8d25"] = "Failed to read the folder.",
        ["\u8def\u5f84\u4e0d\u5b58\u5728\u6216\u4e0d\u662f\u76ee\u5f55"] = "The path does not exist or is not a folder.",
        ["\u8bbe\u8ba1\u8bf4\u660e"] = "Design notes",
        ["\u5c55\u5f00\u8bbe\u8ba1\u8bf4\u660e"] = "Expand design notes",
        ["\u6536\u8d77\u8bbe\u8ba1\u8bf4\u660e"] = "Collapse design notes",
        ["\u8bbe\u8ba1"] = "Design",
        ["\u8bc6\u522b\u56fe\u5c42"] = "Detect layers",
        ["\u5e94\u7528\u751f\u6210"] = "Apply design",
        ["\u8fb9\u7f18\u878d\u5408"] = "Edge blending",
        ["\u95ed\u8fd0\u7b97"] = "Closing",
        ["\u7fbd\u5316"] = "Feather",
        ["\u8fb9\u7f18\u4fdd\u771f"] = "Edge fidelity",
        ["\u672a\u68c0\u6d4b\u5230\u53ef\u7528\u7684\u4eba\u7269\u6216\u5bf9\u8c61\u56fe\u5c42"] = "No usable person or object layers were detected.",
        ["\u5f53\u524d\u6ca1\u6709\u53ef\u5e94\u7528\u7684\u56fe\u5c42"] = "There are no layers to apply.",
        ["\u627e\u4e0d\u5230\u539f\u59cb\u80cc\u666f\u56fe"] = "The original background image was not found.",
        ["\u7f3a\u5c11\u80cc\u666f\u6216\u906e\u7f69\u6570\u636e\uff0c\u8bf7\u5148\u91cd\u65b0\u8bc6\u522b\u56fe\u5c42"] = "Background or mask data is missing. Detect layers again first.",
        ["\u56fe\u5c42\u5408\u6210\u5931\u8d25"] = "Layer composition failed.",
        ["\u8bbe\u8ba1\u5408\u6210"] = "Design composition",
        ["\u8c03\u8282"] = "Adjust",
        ["\u5c55\u5f00\u8c03\u8282"] = "Expand adjustments",
        ["\u6536\u8d77\u8c03\u8282"] = "Collapse adjustments",
        ["\u8c03\u8282\u9762\u677f\u5df2\u6298\u53e0"] = "Adjustment panel collapsed.",
        ["\u8c03\u8282\u9762\u677f\u5df2\u5c55\u5f00"] = "Adjustment panel expanded.",
        ["\u4fdd\u5b58"] = "Save",
        ["\u6e05\u6670"] = "Sharpen",
        ["\u62a0\u56fe"] = "Cutout",
        ["Qwen3.5 \u56fe\u50cf\u5206\u6790"] = "Qwen3.5 image analysis",
        ["\u590d\u5236"] = "Copy",
        ["\u590d\u5236\u5206\u6790\u7ed3\u679c"] = "Copy analysis result",
        ["\u5173\u95ed"] = "Close",
        ["\u5df2\u6709\u56fe\u50cf\u4efb\u52a1\u6b63\u5728\u8fd0\u884c"] = "An image task is already running.",
        ["\u8bf7\u5148\u5728\u5386\u53f2\u8bb0\u5f55\u4e2d\u9009\u62e9\u56fe\u50cf"] = "Select an image from history first.",
        ["\u5206\u6790\u5b8c\u6210"] = "Analysis complete",
        ["\u5206\u6790\u5df2\u53d6\u6d88"] = "Analysis canceled",
        ["\u5206\u6790\u5931\u8d25"] = "Analysis failed",
        ["\u51c6\u5907\u5206\u6790\u5f53\u524d\u5386\u53f2\u56fe\u50cf"] = "Preparing the current history image",
        ["\u6b63\u5728\u52a0\u8f7d\u89c6\u89c9\u6a21\u578b"] = "Loading vision model",
        ["\u6b63\u5728\u7f16\u7801\u56fe\u50cf"] = "Encoding image",
        ["\u6b63\u5728\u52a0\u8f7d\u8bed\u8a00\u6a21\u578b"] = "Loading language model",
        ["\u6b63\u5728\u751f\u6210\u5206\u6790"] = "Generating analysis",
        ["\u6b63\u5728\u6821\u9a8c\u6a21\u578b\u6587\u4ef6"] = "Validating model files",
        ["\u6b63\u5728\u6821\u9a8c\u6a21\u578b\u7ed3\u6784"] = "Validating model structure",
        ["\u6b63\u5728\u52a0\u8f7d\u5206\u8bcd\u5668"] = "Loading tokenizer",
        ["\u6b63\u5728\u5904\u7406\u56fe\u50cf\u4e0e\u63d0\u793a\u8bcd"] = "Processing image and prompt",
        ["\u6b63\u5728\u51c6\u5907 Qwen3.5"] = "Preparing Qwen3.5",
        ["\u6b63\u5728\u53d6\u6d88"] = "Canceling",
        ["\u5206\u6790\u7ed3\u679c\u5df2\u590d\u5236"] = "Analysis result copied.",
        ["\u6e05\u900f"] = "Clear",
        ["\u6696\u8c03"] = "Warm",
        ["\u80f6\u7247"] = "Film",
        ["\u6e05\u65b0"] = "Fresh",
        ["\u4eba\u50cf"] = "Portrait",
        ["\u591c\u666f"] = "Night",
        ["\u5bf9\u6bd4\u5ea6"] = "Contrast",
        ["\u4eae\u5ea6"] = "Brightness",
        ["\u81ea\u7136\u9971\u548c\u5ea6"] = "Vibrance",
        ["\u53bb\u9634\u5f71"] = "Lift shadows",
        ["\u53bb\u9ad8\u5149"] = "Reduce highlights",
        ["\u6696\u8272\u6ee4\u955c"] = "Warm filter",
        ["\u51b7\u8272\u6ee4\u955c"] = "Cool filter",
        ["\u9510\u5316"] = "Sharpen",
        ["\u6a21\u7cca"] = "Blur",
        ["\u53c2\u8003\u56fe"] = "Reference images",
        ["\u7537\u8138"] = "Male face",
        ["\u5973\u8138"] = "Female face",
        ["\u80cc\u666f"] = "Background",
        ["\u6a21\u578b\u63d0\u4f9b\u65b9"] = "Model provider",
        ["\u5e94\u7528\u9884\u8bbe "] = "Applying preset ",
        ["\u9884\u8bbe: "] = "Preset: ",
        ["GPU \u6e05\u6670\u5316"] = "GPU sharpen",
        ["CLIP \u5206\u7c7b"] = "CLIP classification",
        ["\u6362\u8138"] = "Face swap",
        ["\u7f8e\u767d"] = "Whiten",
        ["\u6e05\u6670+\u7f8e\u767d"] = "Sharpen + whiten",
        ["\u6362\u80cc\u666f"] = "Change background",
        ["\u53bb\u96fe+\u8c03\u8272"] = "Dehaze + color grade",
        ["\u8c03\u8272"] = "Color grade",
        ["\u53bb\u96fe"] = "Dehaze",
        ["YOLO \u672a\u68c0\u6d4b\u5230\u4eba\u7269"] = "YOLO found no people",
        ["YOLO \u8bc6\u522b "] = "YOLO detected ",
        ["YOLO\u4fee\u590d "] = "YOLO repaired ",
        ["\u672a\u68c0\u6d4b\u5230\u53ef\u4fee\u590d\u7684\u4eba\u7269\u533a\u57df"] = "No repairable person region was detected.",
        ["\u5f53\u524d\u683c\u5f0f\u6682\u4e0d\u652f\u6301\u8986\u76d6\u4fdd\u5b58"] = "Overwriting this format is not supported.",
        ["\u5df2\u4fdd\u5b58\uff0c\u5e76\u6309\u539f\u8def\u5f84\u91cd\u65b0\u8f7d\u5165"] = "Saved and reloaded from the original path.",
        ["\u5df2\u4fdd\u5b58\u5230\u539f\u8def\u5f84"] = "Saved to the original path.",
        ["\u4fdd\u5b58\u5931\u8d25"] = "Save failed."
    };

    public static AppLanguage CurrentLanguage =>
        PlayerPrefs.GetInt(LanguagePrefKey, (int)AppLanguage.English) == (int)AppLanguage.SimplifiedChinese
            ? AppLanguage.SimplifiedChinese
            : AppLanguage.English;

    public static bool IsEnglish => CurrentLanguage == AppLanguage.English;

    public static void SetLanguage(AppLanguage language)
    {
        PlayerPrefs.SetInt(LanguagePrefKey, (int)language);
        PlayerPrefs.Save();
    }

    public static string Text(string english, string simplifiedChinese)
    {
        return IsEnglish ? english : simplifiedChinese;
    }

    public static string Translate(string text)
    {
        if (!IsEnglish || string.IsNullOrEmpty(text))
            return text;

        if (ChineseToEnglish.TryGetValue(text, out var translated))
            return translated;

        var result = text;
        foreach (var pair in ChineseToEnglish.OrderByDescending(item => item.Key.Length))
            result = result.Replace(pair.Key, pair.Value);
        return result;
    }

    public static void LocalizeVisualTree(VisualElement root)
    {
        if (root == null || !IsEnglish)
            return;

        LocalizeElement(root);
        foreach (var element in root.Query<VisualElement>().ToList())
            LocalizeElement(element);
    }

    private static void LocalizeElement(VisualElement element)
    {
        if (element is TextElement textElement)
            textElement.text = Translate(textElement.text);

        if (!string.IsNullOrWhiteSpace(element.tooltip))
            element.tooltip = Translate(element.tooltip);
    }
}
