#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <stdlib.h>

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
}
