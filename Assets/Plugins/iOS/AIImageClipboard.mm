#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <stdlib.h>

static UIWindow* AIImageActiveWindow()
{
    UIWindow* fallbackWindow = nil;
    if (@available(iOS 13.0, *))
    {
        for (UIScene* scene in UIApplication.sharedApplication.connectedScenes)
        {
            if (scene.activationState != UISceneActivationStateForegroundActive
                || ![scene isKindOfClass:[UIWindowScene class]])
                continue;
            for (UIWindow* candidate in ((UIWindowScene*)scene).windows)
            {
                if (candidate.hidden || candidate.alpha <= 0.0 || candidate.rootViewController == nil)
                    continue;
                if (candidate.isKeyWindow)
                    return candidate;
                if (fallbackWindow == nil || candidate.windowLevel == UIWindowLevelNormal)
                    fallbackWindow = candidate;
            }
        }
    }
    if (fallbackWindow != nil)
        return fallbackWindow;

    UIWindow* keyWindow = UIApplication.sharedApplication.keyWindow;
    return keyWindow != nil && keyWindow.rootViewController != nil ? keyWindow : nil;
}

static UIViewController* AIImageVisibleViewController()
{
    UIWindow* window = AIImageActiveWindow();
    UIViewController* controller = window.rootViewController;
    while (controller != nil)
    {
        if (controller.presentedViewController != nil && !controller.presentedViewController.isBeingDismissed)
        {
            controller = controller.presentedViewController;
            continue;
        }
        if ([controller isKindOfClass:[UINavigationController class]])
        {
            controller = ((UINavigationController*)controller).visibleViewController;
            continue;
        }
        if ([controller isKindOfClass:[UITabBarController class]])
        {
            controller = ((UITabBarController*)controller).selectedViewController;
            continue;
        }
        break;
    }
    return controller;
}

@interface AIImageReportPreviewDelegate : NSObject<UIDocumentInteractionControllerDelegate>
@end

@implementation AIImageReportPreviewDelegate
- (UIViewController*)documentInteractionControllerViewControllerForPreview:(UIDocumentInteractionController*)controller
{
    return AIImageVisibleViewController();
}

- (UIView*)documentInteractionControllerViewForPreview:(UIDocumentInteractionController*)controller
{
    return AIImageVisibleViewController().view;
}

- (CGRect)documentInteractionControllerRectForPreview:(UIDocumentInteractionController*)controller
{
    UIView* view = AIImageVisibleViewController().view;
    return view == nil ? CGRectZero : view.bounds;
}
@end

static AIImageReportPreviewDelegate* AIImageReportPreviewDelegateInstance = nil;
static UIDocumentInteractionController* AIImageReportPreviewController = nil;

static void AIImagePresentRunnerReportShareSheet(NSURL* url, UIViewController* controller)
{
    UIActivityViewController* activityController =
        [[UIActivityViewController alloc] initWithActivityItems:@[url] applicationActivities:nil];
    UIPopoverPresentationController* popover = activityController.popoverPresentationController;
    if (popover != nil)
    {
        popover.sourceView = controller.view;
        popover.sourceRect = controller.view.bounds;
    }
    [controller presentViewController:activityController animated:YES completion:nil];
}

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
            if (controller == nil || controller.view == nil || ![[NSFileManager defaultManager] fileExistsAtPath:value])
            {
                NSLog(@"[Aexis] Cannot present runner report preview: %@", value);
                return;
            }

            if (AIImageReportPreviewDelegateInstance == nil)
                AIImageReportPreviewDelegateInstance = [AIImageReportPreviewDelegate new];
            AIImageReportPreviewController = [UIDocumentInteractionController interactionControllerWithURL:url];
            AIImageReportPreviewController.delegate = AIImageReportPreviewDelegateInstance;
            AIImageReportPreviewController.UTI = @"public.json";
            if ([AIImageReportPreviewController presentPreviewAnimated:YES])
            {
                NSLog(@"[Aexis] Runner report preview opened: %@", value);
            }
            else
            {
                NSLog(@"[Aexis] Runner report preview is unavailable; opening share sheet: %@", value);
                AIImagePresentRunnerReportShareSheet(url, controller);
            }
        });
    }
}
