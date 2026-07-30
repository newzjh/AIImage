#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <stdlib.h>

static UIViewController* AIImageVisibleViewController()
{
    UIWindow* window = nil;
    if (@available(iOS 13.0, *))
    {
        for (UIScene* scene in UIApplication.sharedApplication.connectedScenes)
        {
            if (scene.activationState != UISceneActivationStateForegroundActive
                || ![scene isKindOfClass:[UIWindowScene class]])
                continue;
            for (UIWindow* candidate in ((UIWindowScene*)scene).windows)
            {
                if (candidate.isKeyWindow)
                {
                    window = candidate;
                    break;
                }
            }
            if (window != nil) break;
        }
    }
    if (window == nil)
        window = UIApplication.sharedApplication.keyWindow;

    UIViewController* controller = window.rootViewController;
    while (controller.presentedViewController != nil)
        controller = controller.presentedViewController;
    return controller;
}

@interface AIImageReportPreviewDelegate : NSObject<UIDocumentInteractionControllerDelegate>
@end

@implementation AIImageReportPreviewDelegate
- (UIViewController*)documentInteractionControllerViewControllerForPreview:(UIDocumentInteractionController*)controller
{
    return AIImageVisibleViewController();
}
@end

static AIImageReportPreviewDelegate* AIImageReportPreviewDelegateInstance = nil;
static UIDocumentInteractionController* AIImageReportPreviewController = nil;

extern "C"
{
    void AIImageClipboard_SetText(const char* text)
    {
        NSString* value = text == nullptr ? @"" : [NSString stringWithUTF8String:text];
        [UIPasteboard generalPasteboard].string = value ?: @"";
    }

    const char* AIImageClipboard_GetText()
    {
        NSString* value = [UIPasteboard generalPasteboard].string;
        return value == nil ? nullptr : strdup(value.UTF8String);
    }

    void AIImageClipboard_FreeText(const char* text)
    {
        if (text != nullptr)
            free((void*)text);
    }

    void AIImageReportReveal(const char* path)
    {
        NSString* value = path == nullptr ? @"" : [NSString stringWithUTF8String:path];
        if (value.length == 0) return;

        dispatch_async(dispatch_get_main_queue(), ^{
            NSURL* url = [NSURL fileURLWithPath:value];
            UIViewController* controller = AIImageVisibleViewController();
            if (controller == nil || ![[NSFileManager defaultManager] fileExistsAtPath:value])
                return;

            if (AIImageReportPreviewDelegateInstance == nil)
                AIImageReportPreviewDelegateInstance = [AIImageReportPreviewDelegate new];
            AIImageReportPreviewController = [UIDocumentInteractionController interactionControllerWithURL:url];
            AIImageReportPreviewController.delegate = AIImageReportPreviewDelegateInstance;
            if (![AIImageReportPreviewController presentPreviewAnimated:YES])
            {
                [AIImageReportPreviewController presentOptionsMenuFromRect:controller.view.bounds
                                                                     inView:controller.view
                                                                   animated:YES];
            }
        });
    }
}
