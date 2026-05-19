#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>

extern "C" void AIImageClipboardCopyPNG(const void* data, int length) {
    if (data == NULL || length <= 0) return;
    NSData* d = [NSData dataWithBytes:data length:(NSUInteger)length];
    UIImage* img = [UIImage imageWithData:d];
    if (!img) return;
    UIPasteboard* pb = [UIPasteboard generalPasteboard];
    pb.image = img;
}
#endif

