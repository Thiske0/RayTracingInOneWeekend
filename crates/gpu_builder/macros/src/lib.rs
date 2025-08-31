use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

mod common;

mod derive_device_impl;
#[proc_macro_attribute]
pub fn derive_device_impl(attrs: TokenStream, input: TokenStream) -> TokenStream {
    derive_device_impl::derive_device_impl(attrs, input)
}
mod derive_device_struct;
#[proc_macro_attribute]
pub fn derive_device_struct(attrs: TokenStream, input: TokenStream) -> TokenStream {
    derive_device_struct::derive_device_struct(attrs, input)
}
mod derive_builder;
#[proc_macro_attribute]
pub fn derive_builder(attrs: TokenStream, input: TokenStream) -> TokenStream {
    derive_builder::derive_builder(attrs, input)
}

use crate::common::require_repr;

#[proc_macro_derive(DeviceCopyBuilder)]
pub fn derive_device_copy_builder(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let attributes = &input.attrs;

    if let Err(e) = require_repr(&attributes) {
        return e.to_compile_error().into();
    }

    let name = &input.ident;

    let generics = &input.generics;
    // Process generics (preserve all parameters and where clauses)
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let impl_block = quote! {
        #[cfg(target_os = "cuda")]
        impl #impl_generics gpu_builder::Builder<'_> for #name #ty_generics #where_clause {
            type Output = #name #ty_generics;
        }
        #[cfg(target_os = "cuda")]
        impl #impl_generics gpu_builder::BuildResultType for #name #ty_generics #where_clause {}
    };

    TokenStream::from(impl_block)
}
